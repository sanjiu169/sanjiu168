import pickle, numpy as np
from collections import Counter

class SimpleTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.n_classes = 4
        self.tree = None
    
    def fit(self, X, y, n_classes=4):
        self.n_classes = n_classes
        self.tree = self._grow(X, y, 0)
    
    def _grow(self, X, y, depth):
        n = len(y)
        nc = self.n_classes
        if depth >= self.max_depth or n < 10 or len(np.unique(y)) == 1:
            return {'leaf': True, 'pred': np.bincount(y, minlength=nc) / n}
        best_score = -1
        best_feat, best_thresh = 0, 0
        for feat in range(X.shape[1]):
            vals = X[:, feat]
            thresh = np.median(vals)
            left = vals < thresh
            right = ~left
            if left.sum() < 5 or right.sum() < 5: continue
            score = self._gini(y[left], y[right], n)
            if score > best_score:
                best_score, best_feat, best_thresh = score, feat, thresh
        if best_score <= 0:
            return {'leaf': True, 'pred': np.bincount(y, minlength=nc) / n}
        left_mask = X[:, best_feat] < best_thresh
        return {'leaf': False, 'feature': best_feat, 'threshold': best_thresh,
                'left': self._grow(X[left_mask], y[left_mask], depth+1),
                'right': self._grow(X[~left_mask], y[~left_mask], depth+1)}
    
    def _gini(self, y_left, y_right, total):
        nc = self.n_classes
        def gini(y):
            counts = np.bincount(y, minlength=nc)
            probs = counts / len(y)
            return 1 - np.sum(probs**2)
        return 1 - (len(y_left)*gini(y_left) + len(y_right)*gini(y_right)) / total
    
    def predict_proba(self, X):
        nc = self.n_classes
        probs = np.zeros((len(X), nc))
        for i, x in enumerate(X):
            probs[i] = self._traverse(self.tree, x)
        return probs
    
    def _traverse(self, node, x):
        if node['leaf']: return node['pred']
        if x[node['feature']] < node['threshold']:
            return self._traverse(node['left'], x)
        return self._traverse(node['right'], x)


class GradientBooster:
    def __init__(self, n_trees=100, lr=0.1, max_depth=3):
        self.n_trees = n_trees
        self.lr = lr
        self.max_depth = max_depth
        self.trees = []
        self.init_probs = None
        self.n_classes = 4
    
    def fit(self, X, y, n_classes=None):
        if n_classes is None: n_classes = len(np.unique(y))
        self.n_classes = n_classes
        counts = np.bincount(y, minlength=n_classes)
        self.init_probs = counts / len(y)
        residuals = np.zeros((len(y), n_classes))
        for k in range(n_classes):
            residuals[:, k] = (y == k).astype(float) - self.init_probs[k]
        for t in range(self.n_trees):
            tree = SimpleTree(max_depth=self.max_depth)
            residual_label = np.argmax(residuals, axis=1)
            tree.fit(X, residual_label, n_classes=n_classes)
            self.trees.append(tree)
            preds = tree.predict_proba(X)
            residuals -= self.lr * preds
    
    def predict_proba(self, X):
        n = self.n_classes
        probs = np.zeros((len(X), n))
        probs[:] = self.init_probs[:n]
        for tree in self.trees:
            tp = tree.predict_proba(X)
            if tp.shape[1] < n:
                tp = np.pad(tp, ((0,0),(0,n-tp.shape[1])), constant_values=0.01)
            probs += self.lr * tp[:, :n]
        probs = np.exp(probs - probs.max(axis=1, keepdims=True))
        return probs / probs.sum(axis=1, keepdims=True)


if __name__ == '__main__':
    print("加载数据...")
    with open('engine_memory.pkl', 'rb') as f:
        memory = pickle.load(f)
    data = memory.get('local_history', [])
    print(f"数据总量: {len(data)}期")
    
    def get_type(s):
        if s <= 13: return '小单' if s%2==1 else '小双'
        return '大单' if s%2==1 else '大双'
    
    def get_sum_zone(s):
        zones = {'极小(0-6)':(0,6),'小(7-10)':(7,10),'中(11-16)':(11,16),'大(17-21)':(17,21),'极大(22-27)':(22,27)}
        for name,(lo,hi) in zones.items():
            if lo<=s<=hi: return name
        return '中(11-16)'
    
    def extract_features(item):
        nums = item['nums']; s = item['sum']
        return {'sum':s,'sum_zone':get_sum_zone(s),'type':item['type'],
                'span':max(nums)-min(nums),'tail':s%10,
                'odd_count':sum(1 for n in nums if n%2==1),
                'has_dup':len(set(nums))<3,'mid_num':sorted(nums)[1],
                'sum_mod3':s%3,'sum_mod5':s%5}
    
    def build_features(history, lookback=10):
        X, y_type, y_zone = [], [], []
        type_map = {'大单':0,'大双':1,'小单':2,'小双':3}
        zone_map = {'极小(0-6)':0,'小(7-10)':1,'中(11-16)':2,'大(17-21)':3,'极大(22-27)':4}
        for i in range(len(history) - lookback - 1):
            window = history[i:i+lookback+1]
            features = []
            for offset in range(lookback):
                item = window[offset]
                features.extend([
                    item['sum']/27.0, item['feat']['span']/27.0,
                    item['feat']['tail']/9.0, item['feat']['odd_count']/3.0,
                    float(item['feat']['has_dup']), item['feat']['mid_num']/27.0,
                    item['feat']['sum_mod3']/2.0, item['feat']['sum_mod5']/4.0,
                    type_map.get(item['type'],0)/3.0
                ])
            target = window[0]
            X.append(features)
            y_type.append(type_map[target['type']])
            y_zone.append(zone_map[target['sum_zone']])
        return np.array(X, dtype=np.float32), np.array(y_type), np.array(y_zone)
    
    print("构建特征...")
    X, y_type, y_zone = build_features(data, lookback=10)
    print(f"样本数: {len(X)}, 特征数: {X.shape[1]}")
    
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    split = int(len(X) * 0.8)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_type_train, y_type_test = y_type[idx[:split]], y_type[idx[split:]]
    y_zone_train, y_zone_test = y_zone[idx[:split]], y_zone[idx[split:]]
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    print("\n训练类型预测模型...")
    model_type = GradientBooster(n_trees=30, lr=0.05, max_depth=3)
    model_type.fit(X_train, y_type_train, n_classes=4)
    
    print("训练区间预测模型...")
    model_zone = GradientBooster(n_trees=30, lr=0.05, max_depth=3)
    model_zone.fit(X_train, y_zone_train, n_classes=5)
    
    probs_type = model_type.predict_proba(X_test)
    type_acc = np.mean(np.argmax(probs_type, axis=1) == y_type_test)
    probs_zone = model_zone.predict_proba(X_test)
    zone_acc = np.mean(np.argmax(probs_zone, axis=1) == y_zone_test)
    
    print(f"\n测试集准确率:")
    print(f"类型预测: {type_acc:.1%}")
    print(f"区间预测: {zone_acc:.1%}")
    
    model_data = {
        'model_type': model_type, 'model_zone': model_zone,
        'type_acc': type_acc, 'zone_acc': zone_acc,
        'feature_count': X.shape[1]
    }
    with open('xgboost_models.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("\n模型已保存!")
