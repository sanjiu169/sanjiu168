import pickle, hashlib, random, math
from collections import defaultdict, Counter

TYPES = ['大单','大双','小单','小双']
TYPE_NUMS = {'大单':[15,17,19,21,23,25,27],'大双':[14,16,18,20,22,24,26],'小单':[1,3,5,7,9,11,13],'小双':[0,2,4,6,8,10,12]}
OPPOSITE = {'大单':'小双','小双':'大单','大双':'小单','小单':'大双'}
Y3 = {0:[0,3,6,9,12,15,18,21,24,27],1:[1,4,7,10,13,16,19,22,25],2:[2,5,8,11,14,17,20,23,26]}

def feat_get_combo(history): return [item["type"] for item in history]
def feat_get_5y(history):
    seq=['0/0']
    for i in range(1,len(history)):
        d=history[i]['sum']-history[i-1]['sum']
        seq.append(f"{(d%5+5)%5}/{(history[i-1]['sum']+history[i]['sum'])%5}")
    return seq
def feat_get_diff_sign(history):
    seq=['0']
    for i in range(1,len(history)):
        d=history[i]['sum']-history[i-1]['sum']
        seq.append('正' if d>0 else ('负' if d<0 else '零'))
    return seq
def feat_get_diff_level(history):
    seq=['S']
    for i in range(1,len(history)):
        d=abs(history[i]['sum']-history[i-1]['sum'])
        seq.append('S' if d<=2 else ('M' if d<=6 else 'L'))
    return seq
def feat_get_size_streak(history):
    seq=['1']
    for i in range(1,len(history)):
        cnt=1
        for j in range(i,0,-1):
            if (history[j]['sum']>13)==(history[j-1]['sum']>13): cnt+=1
            else: break
        seq.append(str(min(cnt,5)))
    return seq
def feat_get_parity_streak(history):
    seq=['1']
    for i in range(1,len(history)):
        cnt=1
        for j in range(i,0,-1):
            if (history[j]['sum']%2)==(history[j-1]['sum']%2): cnt+=1
            else: break
        seq.append(str(min(cnt,5)))
    return seq
def feat_get_3y_size(history):
    def y3(s):
        for y,vals in Y3.items():
            if s in vals: return y
        return s%3
    return [f"{y3(item['sum'])}|{'大' if item['sum']>13 else '小'}" for item in history]
def feat_get_prime(history):
    def ip(s):
        if s<=1: return '特殊'
        if s==2: return '质'
        for i in range(2,int(s**0.5)+1):
            if s%i==0: return '合'
        return '质'
    return [ip(item['sum']) for item in history]
def feat_get_volatility(history):
    seq=['平稳']*3
    for i in range(3,len(history)):
        r3=[history[j]['sum'] for j in range(i-2,i+1)]
        avg=sum(r3)/3;var=sum((s-avg)**2 for s in r3)/3
        seq.append('高波动' if var>15 else ('中波动' if var>5 else '平稳'))
    return seq
def feat_get_tail(history):
    return [str(item['sum']%10) for item in history]
def feat_get_slope(history):
    seq=['平稳']*2
    for i in range(2,len(history)):
        slope=history[i]['sum']-history[i-2]['sum']
        seq.append('上升' if slope>1 else ('下降' if slope<-1 else '平稳'))
    return seq

FEATURES = [
    ('原始开奖', feat_get_combo),
    ('5Y基础', feat_get_5y),
    ('差值正负', feat_get_diff_sign),
    ('差值级别', feat_get_diff_level),
    ('连续大小', feat_get_size_streak),
    ('连续单双', feat_get_parity_streak),
    ('3Y大小', feat_get_3y_size),
    ('和值质合', feat_get_prime),
    ('波动率', feat_get_volatility),
    ('和值尾数', feat_get_tail),
    ('走势斜率', feat_get_slope),
]

class PatternMemory:
    def __init__(self, decay=0.85, min_samples=3):
        self.memory = {}
        self.decay = decay
        self.min_samples = min_samples
    
    def _hash(self, seq):
        return hashlib.md5('|'.join(seq).encode()).hexdigest()[:12]
    
    def train(self, data, feature_idx, window):
        feat_func = FEATURES[feature_idx][1]
        for i in range(len(data)-window-1):
            wd = data[i:i+window]
            seq = feat_func(wd)[-window:]
            key = self._hash(seq)
            target = data[i+window]
            tw = self.decay ** (len(data)-i)
            
            if key not in self.memory:
                self.memory[key] = {'hits':Counter(), 'weighted':Counter(), 'total':0, 'total_weight':0}
            rec = self.memory[key]
            rec['hits'][target['type']] += 1
            rec['weighted'][target['type']] += tw
            rec['total'] += 1
            rec['total_weight'] += tw
    
    def predict(self, current_data, feature_idx, window):
        feat_func = FEATURES[feature_idx][1]
        seq = feat_func(current_data)[-window:]
        key = self._hash(seq)
        rec = self.memory.get(key)
        if not rec or rec['total'] < self.min_samples:
            return None, 0
        worst = min(TYPES, key=lambda t: rec['weighted'].get(t,0)/max(0.001,rec['total_weight']))
        wr = rec['weighted'].get(worst,0)/max(0.001,rec['total_weight'])
        return worst, (1-wr)*100
    
    def add_one(self, seq, target_type):
        key = self._hash(seq)
        if key not in self.memory:
            self.memory[key] = {'hits':Counter(), 'weighted':Counter(), 'total':0, 'total_weight':0}
        rec = self.memory[key]
        rec['hits'][target_type] += 1
        rec['weighted'][target_type] += 1
        rec['total'] += 1
        rec['total_weight'] += 1

class MetaLearner:
    def __init__(self):
        self.state_memory = {}
    
    def get_state_key(self, data):
        if len(data)<8: return 'default'
        td=data[:8]
        dragon=1
        for i in range(1,min(8,len(td))):
            if td[i]['type']==td[0]['type']: dragon+=1
            else: break
        alt_size=sum(1 for i in range(1,min(8,len(td))) if (td[i]['sum']>13)!=(td[i-1]['sum']>13))
        miss={}
        for c in TYPES:
            m=0
            for i,item in enumerate(td):
                if item['type']==c: m=i;break
            miss[c]=m
        return f"d{min(dragon,5)}_s{alt_size}_m{max(miss.values())}"
    
    def record(self, state_key, feature_idx, window, is_correct):
        fk = f"{feature_idx}_{window}"
        if state_key not in self.state_memory:
            self.state_memory[state_key] = {}
        if fk not in self.state_memory[state_key]:
            self.state_memory[state_key][fk] = {'wins':0,'total':0}
        rec = self.state_memory[state_key][fk]
        rec['total'] += 1
        if is_correct: rec['wins'] += 1
    
    def get_best(self, state_key):
        if state_key not in self.state_memory: return None
        scores=[]
        for fk,rec in self.state_memory[state_key].items():
            if rec['total']>=3:
                fi,w=fk.split('_')
                scores.append((int(fi),int(w),rec['wins']/rec['total']))
        if not scores: return None
        scores.sort(key=lambda x:x[2],reverse=True)
        return scores[0]

# ===== 训练 =====
if __name__=='__main__':
    try:
        with open('engine_memory.pkl','rb') as f:
            mem=pickle.load(f)
        data=mem.get('local_history',[])
        data.sort(key=lambda x:int(x['expect']))
        print(f"加载{len(data)}期")
    except:
        data=[];print("无数据")
    
    if len(data)<100: exit()
    
    print("训练模式记忆表...")
    pm=PatternMemory(decay=0.85,min_samples=3)
    for fi in range(11):
        for w in range(3,9):
            pm.train(data,fi,w)
    print(f"模式数:{len(pm.memory)}")
    
    print("训练元学习器...")
    ml=MetaLearner();correct=0;total=0
    for i in range(500,len(data)-1):
        past=data[:i];target=data[i]
        if len(past)<8: continue
        state=ml.get_state_key(past[-30:])
        best=ml.get_best(state)
        if best:
            fi,w,_=best
            pred,conf=pm.predict(past[-30:],fi,w)
            if pred:
                is_correct=(pred!=target['type'])
                ml.record(state,fi,w,is_correct)
                total+=1
                if is_correct: correct+=1
    
    if total>0:
        print(f"元学习器: {correct}/{total}={correct/total:.1%}")
    
    model={'pm':pm,'ml':ml}
    with open('v7_model.pkl','wb') as f:
        pickle.dump(model,f)
    print("保存: v7_model.pkl")
