#!/usr/bin/env python
import asyncio, aiohttp, json, math, random, logging, os, pickle, sys
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

sys.path.insert(0, '/data/data/com.termux/files/home')
from train_model import GradientBooster, SimpleTree

BOT_TOKEN = "8027828258:AAFlOfT7fS9F99XQWOUchvBNVynpkBpmXm8"
API_URL = "https://super.pc28998.com/history/JND28?limit=200"
logging.basicConfig(level=logging.INFO)
TYPES = ['大单','大双','小单','小双']
SUM_ZONES = {'极小(0-6)':(0,6),'小(7-10)':(7,10),'中(11-16)':(11,16),'大(17-21)':(17,21),'极大(22-27)':(22,27)}
ODDS = {'大单':3.8,'大双':3.8,'小单':3.8,'小双':3.8}
TYPE_NUMS = {'大单':[15,17,19,21,23,25,27],'大双':[14,16,18,20,22,24,26],'小单':[1,3,5,7,9,11,13],'小双':[0,2,4,6,8,10,12]}

def get_type(s):
    if s<=13: return '小单' if s%2==1 else '小双'
    return '大单' if s%2==1 else '大双'

def get_sum_zone(s):
    for name,(lo,hi) in SUM_ZONES.items():
        if lo<=s<=hi: return name
    return '中(11-16)'

def extract_features(item):
    nums=item['nums'];s=item['sum']
    return {'sum':s,'sum_zone':get_sum_zone(s),'type':item['type'],'span':max(nums)-min(nums),'tail':s%10,'odd_count':sum(1 for n in nums if n%2==1),'has_dup':len(set(nums))<3,'mid_num':sorted(nums)[1],'sum_mod3':s%3,'sum_mod5':s%5}

async def fetch_data():
    timeout=aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.get(API_URL) as r: raw=await r.json()
    if isinstance(raw,dict):
        for k in ['data','items','list','results','records']:
            if isinstance(raw.get(k),list): raw=raw[k];break
    if not isinstance(raw,list): return []
    hist=[]
    for it in raw:
        try:
            code_str=str(it.get('opencode',''))
            nums=[int(x) for x in code_str.split(',') if x.strip().isdigit()]
            if not nums or len(nums)<2: continue
            s=sum(nums)
            item={'expect':str(it.get('expect','')),'nums':nums,'sum':s,'type':get_type(s),'sum_zone':get_sum_zone(s),'opentime':it.get('opentime',0)}
            item['feat']=extract_features(item)
            hist.append(item)
        except: continue
    return hist

class V6Engine:
    def __init__(self):
        self.h=[];self.b={'tc':0.25,'tr':0.25,'hc':0.25,'rw':0.25}
        self.lp=[];self.pl=[];self.model_weights={'markov':0.3,'similar':0.25,'freq':0.25,'feature':0.2}
        self.consecutive_losses=0;self.risk_level='LOW'
        self.number_heat={};self.local_history=[]
        
        # 加载ML模型
        try:
            with open('xgboost_models.pkl','rb') as f:
                ml=pickle.load(f)
            self.ml_model=ml['model_type'];self.ml_zone=ml['model_zone']
            self.ml_type_acc=ml.get('type_acc',0)
            print(f"ML模型已加载 (类型:{self.ml_type_acc:.1%})")
        except Exception as e:
            print(f"ML未加载: {e}")
            self.ml_model=None;self.ml_zone=None
        
        # 加载记忆
        try:
            with open('engine_memory.pkl','rb') as f:
                mem=pickle.load(f)
            self.b=mem.get('b',self.b);self.lp=mem.get('lp',[]);self.pl=mem.get('pl',[])
            self.model_weights=mem.get('model_weights',self.model_weights)
            self.consecutive_losses=mem.get('consecutive_losses',0)
            self.risk_level=mem.get('risk_level','LOW')
            self.local_history=mem.get('local_history',[])
            print(f"记忆加载成功 ({len(self.pl)}期评估, {len(self.local_history)}期历史)")
        except:
            print("无历史记忆")
    
    def save(self):
        try:
            data={'b':self.b,'lp':self.lp[-100:] if self.lp else [],'pl':self.pl,'model_weights':self.model_weights,'consecutive_losses':self.consecutive_losses,'risk_level':self.risk_level,'local_history':self.local_history[:5000]}
            with open('engine_memory.pkl','wb') as f: pickle.dump(data,f)
        except: pass
    
    def merge_history(self,new_data):
        if not hasattr(self,'local_history'): self.local_history=[]
        idx={item['expect']:i for i,item in enumerate(self.local_history)}
        for item in new_data:
            if item['expect'] in idx: self.local_history[idx[item['expect']]]=item
            else: self.local_history.append(item)
        self.local_history.sort(key=lambda x:int(x['expect']),reverse=True)
        self.local_history=self.local_history[:5000]
    
    def ld(self,h):
        self.h=h
        if h: self.merge_history(h)
        if len(self.local_history)>=100: self.h=self.local_history[:120]
    
    def compute_markov(self):
        if len(self.h)<3: return {t:{t2:0.25 for t2 in TYPES} for t in TYPES}
        d=self.h[:80];mx=defaultdict(lambda:defaultdict(float))
        for i in range(len(d)-1): mx[d[i]['type']][d[i+1]['type']]+=1
        prob={}
        for ft in TYPES:
            total=sum(mx[ft].values())
            if total>0:
                prob[ft]={t:max(0.01,mx[ft].get(t,0)/total) for t in TYPES}
                tot=sum(prob[ft].values());prob[ft]={t:v/tot for t,v in prob[ft].items()}
            else: prob[ft]={t:0.25 for t in TYPES}
        return prob
    
    def find_similar_patterns(self,window=5):
        if len(self.h)<window+5: return None
        current=self.h[:window];cv=[]
        for item in current:
            cv.extend([item['sum']/27.0,item['feat']['span']/27.0,item['feat']['tail']/9.0,item['feat']['odd_count']/3.0])
        cv=np.array(cv);similarities=[]
        for i in range(window,len(self.h)-window):
            past=self.h[i:i+window];pv=[]
            for item in past:
                pv.extend([item['sum']/27.0,item['feat']['span']/27.0,item['feat']['tail']/9.0,item['feat']['odd_count']/3.0])
            pv=np.array(pv);sim=1.0/(1.0+np.linalg.norm(cv-pv))
            similarities.append((i,sim,self.h[i-1] if i>0 else self.h[0]))
        similarities.sort(key=lambda x:x[1],reverse=True)
        return similarities[:20]
    
    def bayesian_update(self):
        if len(self.h)<10: return self.b
        rc=self.h[:10];stk=1
        for i in range(1,len(rc)):
            if rc[i]['type']==rc[0]['type']: stk+=1
            else: break
        rev=sum(1 for i in range(1,len(rc)) if rc[i]['type']!=rc[i-1]['type'])
        fq=Counter(it['type'] for it in rc);mh,mc2=max(fq.values()),min(fq.values())
        ent=0
        for t in TYPES:
            p=fq.get(t,0)/len(rc)
            if p>0: ent-=p*math.log2(p)
        ev={'tc':stk/5 if stk<=5 else 1.0,'tr':rev/(len(rc)-1),'hc':1.0 if mh-mc2>=4 else 0.3,'rw':ent/math.log2(4) if len(rc)>1 else 0.5}
        nb={h:ev.get(h,0.5)*self.b[h] for h in self.b}
        tot=sum(nb.values())
        if tot>0: self.b={k:v/tot for k,v in nb.items()}
        return self.b
    
    def predict(self):
        if len(self.h)<5: return None
        markov=self.compute_markov();ct=self.h[0]['type'];mp=markov.get(ct,{t:0.25 for t in TYPES})
        similar=self.find_similar_patterns(5)
        sim_probs={t:0.25 for t in TYPES}
        if similar:
            nt=Counter(s[2]['type'] for s in similar[:20]);tot=sum(nt.values())
            if tot>0:
                for t in TYPES: sim_probs[t]=max(0.02,nt.get(t,0)/tot)
                tot=sum(sim_probs.values());sim_probs={t:v/tot for t,v in sim_probs.items()}
        fq=Counter(item['type'] for item in self.h[:30]);tot=sum(fq.values());fp={t:fq.get(t,0)/tot for t in TYPES}
        mis={};[mis.setdefault(t,0) for t in TYPES]
        for t in TYPES:
            c=0
            for it in self.h:
                if it['type']==t: break
                c+=1
            mis[t]=c
        feat_pred={t:0.25 for t in TYPES}
        if len(self.h)>10:
            lf=self.h[0]['feat']
            for t in TYPES:
                sc,ctr=0,0
                for i in range(1,min(30,len(self.h))):
                    if self.h[i]['type']==t:
                        if abs(lf['span']-self.h[i-1]['feat']['span'])<3 and abs(lf['tail']-self.h[i-1]['feat']['tail'])<2: sc+=1
                        ctr+=1
                feat_pred[t]=(sc+1)/(ctr+4) if ctr>0 else 0.25
        
        # 动态权重
        if self.pl and len(self.pl)>=3:
            recent=[r['hit'] for r in self.pl[-10:]];hr=sum(recent)/len(recent)
            if hr>0.6:
                self.model_weights['markov']=min(0.5,self.model_weights['markov']+0.02)
                self.model_weights['similar']=max(0.1,self.model_weights['similar']-0.01)
            elif hr<0.4:
                self.model_weights['similar']=min(0.5,self.model_weights['similar']+0.02)
                self.model_weights['markov']=max(0.1,self.model_weights['markov']-0.01)
            tot_w=sum(self.model_weights.values());self.model_weights={k:v/tot_w for k,v in self.model_weights.items()}
        
        mw=self.model_weights;bf=self.bayesian_update()
        
        # 各模型独立预测
        model_preds={
            'markov':sorted(TYPES,key=lambda t:mp.get(t,0),reverse=True)[0],
            'similar':sorted(TYPES,key=lambda t:sim_probs.get(t,0),reverse=True)[0],
            'freq':sorted(TYPES,key=lambda t:fp.get(t,0),reverse=True)[0],
            'feature':sorted(TYPES,key=lambda t:feat_pred.get(t,0),reverse=True)[0]
        }
        votes={t:0 for t in TYPES}
        for model,pred in model_preds.items(): votes[pred]+=3*mw.get(model,0.25)
        
        # 手工融合
        risk_factor={'LOW':1.0,'MEDIUM':0.8,'HIGH':0.6,'CRITICAL':0.4};rf=risk_factor.get(self.risk_level,0.8)
        fu={}
        for t in TYPES:
            fu[t]=(mw['markov']*mp.get(t,0.25)+mw['similar']*sim_probs.get(t,0.25)+mw['freq']*fp.get(t,0)+mw['feature']*feat_pred.get(t,0.25)+bf['tr']*(1.0 if mis[t]>=6 else 0))*rf
        
        # ML预测
        ml_probs=None
        try:
            if self.ml_model is not None and len(self.h)>=10:
                mf=[]
                for offset in range(10):
                    if offset<len(self.h):
                        item=self.h[offset]
                        mf.extend([item['sum']/27.0,item['feat']['span']/27.0,item['feat']['tail']/9.0,item['feat']['odd_count']/3.0,int(item['feat']['has_dup']),item['feat']['mid_num']/27.0,item['feat']['sum_mod3']/2.0,item['feat']['sum_mod5']/4.0,{'大单':0,'大双':1,'小单':2,'小双':3}.get(item['type'],0)/3.0])
                    else: mf.extend([0.0]*9)
                X_ml=np.array([mf[:90]],dtype=np.float32);raw=self.ml_model.predict_proba(X_ml)[0]
                ml_probs={'大单':raw[0],'大双':raw[1],'小单':raw[2],'小双':raw[3]}
        except: pass
        
        # 三合一融合: 手工35% + 投票35% + ML30%
        # 分歧大时降ML权重，提高投票
        vote_max = max(votes.values()) / max(1,sum(votes.values()))
        if vote_max > 0.4:  # 模型意见统一
            ml_weight = 0.3
        else:  # 分歧大
            ml_weight = 0.1
        
        for t in TYPES:
            fu[t]=fu[t]*0.4+votes[t]/max(1,sum(votes.values()))*0.4
        if ml_probs:
            for t in TYPES: fu[t]+=ml_probs.get(t,0.25)*ml_weight
        
        tot2=sum(fu.values());fu={t:v/tot2 for t,v in fu.items()}
        st=sorted(fu.items(),key=lambda x:x[1],reverse=True)
        gap=st[0][1]-st[1][1];cf='HIGH' if gap>0.2 else ('MEDIUM' if gap>0.1 else 'LOW')
        
        # 区间预测
        zf=Counter(item['sum_zone'] for item in self.h[:30]);zt=sum(zf.values());zp={z:zf.get(z,0)/zt for z in SUM_ZONES}
        zs=sorted(zp.items(),key=lambda x:x[1],reverse=True)
        
        kl=st[0][0];db=[st[1][0],st[2][0]]
        tm=[]
        for dt in db[:2]:
            ns=TYPE_NUMS[dt];nwm=[]
            for n in ns:
                miss=0
                for it in self.h:
                    if it['sum']==n: break
                    miss+=1
                nwm.append((n,miss))
            nwm.sort(key=lambda x:x[1],reverse=True)
            for n,_ in nwm[:2]:
                if n not in tm: tm.append(n)
        tm=sorted(tm[:4])
        av=st[-1][0] if st[-1][1]<0.12 else '无'
        best_zone=zs[0][0] if zs else '中(11-16)';zr=SUM_ZONES.get(best_zone,(11,16))
        next_expect=str(int(self.h[0]['expect'])+1) if self.h and self.h[0]['expect'].isdigit() else '--'
        
        pred={
            'expect':self.h[0]['expect'] if self.h else '--','next_expect':next_expect,
            'kl':kl,'db':f"{db[0]}+{db[1]}",'tm':', '.join(f"{n:02d}" for n in tm),
            'cf':cf,'gap':f"{gap:.1%}",'pr':{t:f"{p:.1%}" for t,p in fu.items()},
            'zone':best_zone,'zone_range':f"{zr[0]}-{zr[1]}",
            'zone_probs':{z:f"{p:.1%}" for z,p in zp.items()},
            'av':av,'bf':{k:f"{v:.1%}" for k,v in bf.items()},
            'mw':{k:f"{v:.1%}" for k,v in mw.items()},
            'similar_count':len(similar) if similar else 0,
            'risk_level':self.risk_level,'consecutive_losses':self.consecutive_losses,
            'ts':(datetime.now()+timedelta(hours=8)).strftime('%m-%d %H:%M:%S')
        }
        self.lp.append(pred)
        if len(self.lp)>100: self.lp.pop(0)
        return pred

engine=V6Engine()

async def start(u,c):
    await u.message.reply_text("🎯 三九V6.0 ML版\n/predict /evaluate /accuracy /beliefs /risk /heat")

async def predict(u,c):
    msg=await u.message.reply_text('🔄 V6.0+ML计算中...')
    try:
        h=await fetch_data();engine.ld(h)
        p=engine.predict()
        if p:
            rl={'LOW':'✅低风险','MEDIUM':'⚠️中风险','HIGH':'🔴高风险','CRITICAL':'🚫极高风险'}
            cm={'HIGH':'🟢高','MEDIUM':'🟡中','LOW':'🔴低'}
            txt=f"""🎯 三九V6.0+ML

📋 第{p['next_expect']}期 (上期{p['expect']})
{rl.get(p['risk_level'],'')} | 连败:{p['consecutive_losses']}期

📊 类型: 杀{p['kl']} | {p['db']}
🎯 特码: {p['tm']}
📦 区间: {p['zone']}({p['zone_range']})

{cm.get(p['cf'],p['cf'])}置信度({p['gap']})

📈 类型概率: 大单{p['pr']['大单']} 大双{p['pr']['大双']} 小单{p['pr']['小单']} 小双{p['pr']['小双']}
⚠️ 避开: {p.get('av','无')} | 相似段:{p['similar_count']}
🕐 {p['ts']}"""
            await msg.edit_text(txt)
        else: await msg.edit_text('数据不足')
    except Exception as e: await msg.edit_text(f'错误: {str(e)[:50]}')

async def evaluate(u,c):
    if engine.lp and len(engine.h)>0:
        lp=engine.lp[-1];ac=engine.h[0]['type']
        kh=ac!=lp['kl'];dt=lp['db'].split('+');dh=ac in [x.strip() for x in dt]
        hit=kh or dh;engine.pl.append({'kl':lp['kl'],'db':lp['db'],'ac':ac,'hit':hit})
        engine.save()
        st='✅命中' if hit else '❌未中'
        rc=engine.pl[-10:];a10=sum(1 for r in rc if r['hit'])/len(rc) if rc else 0
        await u.message.reply_text(f"📋 上期: 预测{lp['kl']}|{lp['db']} 实际{ac} {st}\n近10期:{a10:.0%}")
    else: await u.message.reply_text('暂无记录')

async def accuracy(u,c):
    if not engine.pl: await u.message.reply_text('暂无数据');return
    t=len(engine.pl);h=sum(1 for r in engine.pl if r['hit'])
    a10=sum(1 for r in engine.pl[-10:] if r['hit'])/min(10,t)
    await u.message.reply_text(f"📊 总{t}期|命中{h}({h/t:.1%})|近10:{a10:.1%}")

async def beliefs(u,c):
    b=engine.b;mw=engine.model_weights
    await u.message.reply_text(f"🧠 信念: 续{b['tc']:.1%} 反{b['tr']:.1%} 冷{b['hc']:.1%} 随{b['rw']:.1%}\n📊 权重: 马{mw['markov']:.1%} 似{mw['similar']:.1%} 频{mw['freq']:.1%} 特{mw['feature']:.1%}")

async def risk(u,c):
    rl={'LOW':'✅低','MEDIUM':'⚠️中','HIGH':'🔴高','CRITICAL':'🚫极高'}
    await u.message.reply_text(f"🛡 风险:{rl.get(engine.risk_level,'?')} | 连败:{engine.consecutive_losses}期")

async def broadcast_on(u,c):
    # global removed
    BROADCAST_CHAT_ID = u.message.chat_id; open("chat_id.txt","w").write(str(BROADCAST_CHAT_ID))
    with open('chat_id.txt','w') as f: f.write(str(BROADCAST_CHAT_ID))
    await u.message.reply_text("✅ 自动播报已开启")

async def broadcast_off(u,c):
    # global removed
    cid2 = None
    with open('chat_id.txt','w') as f: f.write('')
    await u.message.reply_text("❌ 自动播报已关闭")

async def heat(u,c):
    if not engine.number_heat: await u.message.reply_text('暂无');return
    sh=sorted(engine.number_heat.items(),key=lambda x:x[1].get('heat_score',0),reverse=True)
    hot=' '.join(f"{n:02d}" for n,_ in sh[:5]);cold=' '.join(f"{n:02d}" for n,_ in sh[-5:])
    await u.message.reply_text(f"🔥热号: {hot}\n❄冷号: {cold}")

async def get_next_open_time():
    try:
        timeout=aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(API_URL) as r: raw=await r.json()
        if isinstance(raw,dict):
            for k in ['data','items','list']:
                if isinstance(raw.get(k),list): raw=raw[k];break
        if not raw or not isinstance(raw,list): return None
        ts=raw[0].get('opentime',0)
        if isinstance(ts,str):
            try: ts=int(ts)
            except: return None
        return datetime.fromtimestamp(ts)+timedelta(seconds=210)
    except: return None

async def auto_learn():
    print("🧠 V6.0+ML 自动学习启动")
    while True:
        try:
            h=await fetch_data();engine.ld(h)
            if engine.lp and len(engine.h)>0:
                lp=engine.lp[-1];ac=engine.h[0]['type']
                kh=ac!=lp['kl'];dt=lp['db'].split('+');dh=ac in [x.strip() for x in dt]
                engine.pl.append({'kl':lp['kl'],'db':lp['db'],'ac':ac,'hit':kh or dh})
                engine.save()
                print(f"{'✅' if (kh or dh) else '❌'} {lp['kl']} vs {ac}")
            p=engine.predict()
            if p:
                print(f"📊 {p['next_expect']}|{p['kl']}|{p['zone']}|{p['cf']}")
                # 每5期自动推送一次，保持Render不休眠
                try:
                    if int(p['next_expect']) % 5 == 0:
                        # 推送到自己（需要存chat_id）
                        pass
                except: pass
            nt=await get_next_open_time()
            if nt:
                wait=(nt-datetime.now()).total_seconds()+8
                if 0<wait<600: await asyncio.sleep(wait)
                else: await asyncio.sleep(10)
            else: await asyncio.sleep(210)
        except Exception as e:
            print(f"错误:{e}")
            await asyncio.sleep(60)

def main():
    app=Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler('start',start))
    app.add_handler(CommandHandler('predict',predict))
    app.add_handler(CommandHandler('evaluate',evaluate))
    app.add_handler(CommandHandler('accuracy',accuracy))
    app.add_handler(CommandHandler('beliefs',beliefs))
    app.add_handler(CommandHandler('risk',risk))
    app.add_handler(CommandHandler('broadcast_on',broadcast_on))
    app.add_handler(CommandHandler('broadcast_off',broadcast_off))
    app.add_handler(CommandHandler('heat',heat))
    import threading
    
    # 加载推送目标
    # global removed
    try:
        with open('chat_id.txt','r') as f:
            cid2 = int(f.read().strip())
    except:
        cid2 = None
    
    async def auto_broadcast():
        # global removed
        while True:
            try:
                if cid2:
                    await app.bot.send_message(cid, "🔄 心跳检测，保持在线")
                await asyncio.sleep(840)  # 14分钟一次
            except:
                await asyncio.sleep(60)
    
    t=threading.Thread(target=lambda:(asyncio.new_event_loop().run_until_complete(auto_learn())),daemon=True)
    t.start()
    t2=threading.Thread(target=lambda:(asyncio.new_event_loop().run_until_complete(auto_broadcast())),daemon=True)
    t2.start()
    
    print("🤖 V6.0+ML就绪")
    
    # 启动极简HTTP服务器（避免Render超时）
    from aiohttp import web
    async def health(request):
        return web.Response(text="OK")
    http_app = web.Application()
    http_app.router.add_get('/', health)
    
    import os
    port = int(os.getenv('PORT', 8080))
    
    async def run_http():
        runner = web.AppRunner(http_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        print(f"HTTP健康检查端口:{port}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(run_http())
    app.run_polling()

if __name__=='__main__':
    main()
