#!/usr/bin/env python3
# 第 47 期玩具实验：24 点上的搜索
# 单文件版：下面依次是极小的自动微分、迷你 Transformer 和实验本身，只依赖 NumPy。
import numpy as np
class T:
    def __init__(s,v,ps=(),bk=None):
        s.v=np.asarray(v,dtype=np.float64); s.g=None; s.ps=ps; s.bk=bk
    def __add__(s,o):
        o=o if isinstance(o,T) else T(o); r=T(s.v+o.v,(s,o))
        r.bk=lambda g:(unb(g,s.v.shape),unb(g,o.v.shape)); return r
    def __mul__(s,o):
        o=o if isinstance(o,T) else T(o); r=T(s.v*o.v,(s,o))
        r.bk=lambda g:(unb(g*o.v,s.v.shape),unb(g*s.v,o.v.shape)); return r
    def __matmul__(s,o):
        r=T(s.v@o.v,(s,o))
        r.bk=lambda g:(unb(g@np.swapaxes(o.v,-1,-2),s.v.shape),unb(np.swapaxes(s.v,-1,-2)@g,o.v.shape)); return r
    def t(s,*ax):
        r=T(s.v.transpose(ax),(s,)); inv=np.argsort(ax); r.bk=lambda g:(g.transpose(inv),); return r
    def rs(s,*sh):
        r=T(s.v.reshape(sh),(s,)); old=s.v.shape; r.bk=lambda g:(g.reshape(old),); return r
    def relu(s):
        m=s.v>0; r=T(s.v*m,(s,)); r.bk=lambda g:(g*m,); return r
    def softmax(s):
        x=s.v-s.v.max(-1,keepdims=True); e=np.exp(x); p=e/e.sum(-1,keepdims=True); r=T(p,(s,))
        def bk(g):
            gp=(g*p).sum(-1,keepdims=True); return (p*(g-gp),)
        r.bk=bk; return r
    def lnorm(s,eps=1e-6):
        # 只算方差归一化（无可学习缩放）
        m=s.v.mean(-1,keepdims=True); d=s.v-m; sd=np.sqrt((d*d).mean(-1,keepdims=True)+eps); r=T(d/sd,(s,))
        def bk(g):
            gc=g-g.mean(-1,keepdims=True); y=d/sd; gy=(g*y).mean(-1,keepdims=True)
            return ((gc-gy*y)/sd,)
        r.bk=bk; return r
    def backward(s,seed=None):
        order=[];seen=set()
        def go(n):
            if id(n) in seen:return
            seen.add(id(n))
            for p in n.ps: go(p)
            order.append(n)
        go(s); s.g=np.ones_like(s.v) if seed is None else seed
        for n in reversed(order):
            if n.bk is None or n.g is None: continue
            for p,g in zip(n.ps,n.bk(n.g)):
                p.g=g if p.g is None else p.g+g
def unb(g,sh):
    while g.ndim>len(sh): g=g.sum(0)
    for i,d in enumerate(sh):
        if d==1 and g.shape[i]!=1: g=g.sum(i,keepdims=True)
    return g
class Adam:
    def __init__(s,ps,b1=.9,b2=.98,eps=1e-9):
        s.ps=ps;s.m=[np.zeros_like(p.v) for p in ps];s.u=[np.zeros_like(p.v) for p in ps];s.b1,s.b2,s.eps=b1,b2,eps;s.n=0
    def step(s,lr):
        s.n+=1
        for i,p in enumerate(s.ps):
            if p.g is None: continue   # 本步没用到的参数不更新
            s.m[i]=s.b1*s.m[i]+(1-s.b1)*p.g; s.u[i]=s.b2*s.u[i]+(1-s.b2)*p.g*p.g
            p.v-=lr*(s.m[i]/(1-s.b1**s.n))/(np.sqrt(s.u[i]/(1-s.b2**s.n))+s.eps); p.g=None

import numpy as np
class Enc:
    def __init__(s,V,L,d=32,h=4,ff=64,N=2,seed=0,causal=False,n_cls=0,pre=False):
        r=np.random.default_rng(seed); s.V,s.L,s.d,s.h,s.ff,s.N,s.causal,s.pre=V,L,d,h,ff,N,causal,pre; s.ps=[]
        s.E=s.par(r.normal(0,d**-.5,(V,d))); s.P=s.par(r.normal(0,d**-.5,(L,d)))
        lin=lambda a,b:s.par(r.normal(0,1/np.sqrt(a),(a,b)))
        s.layers=[dict(q=lin(d,d),k=lin(d,d),v=lin(d,d),o=lin(d,d),a=lin(d,ff),ab=s.par(np.zeros(ff)),b=lin(ff,d),bb=s.par(np.zeros(d))) for _ in range(N)]
        if pre:   # GPT-2：残差层权重初始化乘 1/√(残差层数)
            for l in s.layers:
                l['o'].v*=1/np.sqrt(2*N); l['b'].v*=1/np.sqrt(2*N)
        s.head=None; s.rng=r
    def par(s,v): t=T(v); s.ps.append(t); return t
    def add_head(s,k):
        s.head=T(s.rng.normal(0,1/np.sqrt(s.d),(s.d,k))); s.ps.append(s.head); return s.head
    def attn(s,l,x):
        B,n,_=x.v.shape; h,dk=s.h,s.d//s.h
        sp=lambda t:t.rs(B,n,h,dk).t(0,2,1,3)
        Q,K,W=sp(x@l['q']),sp(x@l['k']),sp(x@l['v'])
        sc=Q@K.t(0,1,3,2)*(1/np.sqrt(dk))
        if s.causal:
            mask=np.tril(np.ones((n,n),bool)); o=sc; sc=T(np.where(mask,o.v,-1e9),(o,)); sc.bk=lambda g,mask=mask:(g*mask,)
        a=sc.softmax(); s.last=a.v
        return (a@W).t(0,2,1,3).rs(B,n,h*dk)@l['o']
    def encode(s,x):
        n=x.shape[1]; h=T(np.eye(s.V)[x])@s.E*np.sqrt(s.d)+T(np.eye(s.L)[:n])@s.P
        for l in s.layers:
            if s.pre:
                h=h+s.attn(l,h.lnorm()); g=h.lnorm(); h=h+(((g@l['a'])+l['ab']).relu()@l['b'])+l['bb']
            else:
                h=(h+s.attn(l,h)).lnorm(); h=(h+(((h@l['a'])+l['ab']).relu()@l['b'])+l['bb']).lnorm()
        return h.lnorm() if s.pre else h
    def lm(s,x): return s.encode(x)@s.E.t(1,0)
def ce(lg,y,w=None):
    """交叉熵。w 是每个位置的权重（如只算被遮的词）。返回 损失值, 概率节点, 反向种子"""
    p=lg.softmax(); idx=np.indices(y.shape); py=p.v[(*idx,y)]
    if w is None: w=np.ones(y.shape)
    W=w.sum()+1e-9; val=-(w*np.log(py+1e-12)).sum()/W
    sd=np.zeros(p.v.shape); sd[(*idx,y)]=-w/(py+1e-12)/W
    return val,p,sd
def noam(st,d,warm): return d**-.5*min(st**-.5,st*warm**-1.5)

"""第 47 期玩具实验：24 点上的 ToT。没有语言模型。用很小的 MLP 代替：生成器（给每个合法下一步打分）从少量解题记录模仿学习，
评估器（判断状态能否到 24）从少量带标签的状态学习。两者都故意只学一点点，所以都有错。
比较：一条路走到底（类 CoT）、多次多数投票（类 CoT-SC）、最好的 k 次（有答案作弊）、BFS（类 ToT）。这是玩具，不代表论文结果。"""
import sys,json,time,math,itertools,numpy as np
from fractions import Fraction as Fr
OPS=['+','-','*','/']
def moves(s):
    """状态是数字列表，返回所有下一步 (i,j,op,结果状态)"""
    out=[]
    for i,j in itertools.permutations(range(len(s)),2):
        a,b=s[i],s[j];rest=[s[k] for k in range(len(s)) if k not in(i,j)]
        for op in OPS:
            if op in '+*' and i>j:continue
            if op=='/' and b==0:continue
            c=a+b if op=='+' else a-b if op=='-' else a*b if op=='*' else a/b
            out.append((a,b,op,sorted(rest+[c])))
    return out
MEMO={}
def solvable(s):
    k=tuple(s)
    if k in MEMO:return MEMO[k]
    if len(s)==1:
        v=s[0]==24
    else:
        v=any(solvable(m[3]) for m in moves(s))
    MEMO[k]=v;return v
def count_paths(s):
    if len(s)==1:return 1 if s[0]==24 else 0
    return sum(count_paths(m[3]) for m in moves(s))
def feat(s):
    """状态特征：最多 4 个数；再加“看一步”的线索：只剩 1 个数时离 24 多远，剩 2 个数时一步能多接近 24。类似论文里评估器做的少量试算"""
    f=np.zeros(4*5+3+3)
    for i,v in enumerate(s):
        x=float(v);f[i*5:i*5+5]=(1,min(abs(x),200)/50,math.log1p(abs(x))/3,1. if x<0 else 0.,1. if v.denominator==1 else 0.)
    f[20+min(len(s),3)-1]=1
    if len(s)==1:
        f[23]=1. if s[0]==24 else 0.;f[24]=math.exp(-abs(float(s[0])-24)/4)
    elif len(s)==2:
        a,b=s;c=[a+b,a-b,b-a,a*b]+([a/b] if b!=0 else [])+([b/a] if a!=0 else []);d=min(abs(float(x)-24) for x in c);f[25]=math.exp(-d/2)
    else:
        a=[float(v) for v in s];f[25]=math.exp(-min(abs(a[0]*a[1]-24),abs(a[1]*a[2]-24),abs(a[0]*a[2]-24),abs(sum(a)-24))/8)
    return f
class MLP:
    def __init__(s,di,H,seed):
        r=np.random.default_rng(seed);s.W=[r.normal(0,1/np.sqrt(di),(di,H)),r.normal(0,1/np.sqrt(H),(H,H)),r.normal(0,.01,(H,1))];s.b=[np.zeros(H),np.zeros(H),np.zeros(1)]
    def fw(s,x):
        hs=[x];h=x
        for i in range(3):
            y=h@s.W[i]+s.b[i]
            if i<2:y=np.maximum(y,0)
            hs.append(y);h=y
        return h[:,0],hs
    def train(s,X,grad,steps,seed,B=64,lr=2e-3):
        """grad(idx,out)->每个样本的 dL/dout"""
        r=np.random.default_rng(seed);P=s.W+s.b;M=[np.zeros_like(p) for p in P];U=[np.zeros_like(p) for p in P]
        for st in range(1,steps+1):
            i=r.integers(0,len(X),B);o,hs=s.fw(X[i]);u=grad(i,o)[:,None];G=[None]*6;G[2]=hs[2].T@u/B;G[5]=u.mean(0);u=u@s.W[2].T
            for k in (1,0):
                u=u*(hs[k+1]>0);G[k]=hs[k].T@u/B;G[3+k]=u.mean(0);u=u@s.W[k].T
            for k,(q,g) in enumerate(zip(P,G)):
                M[k]=.9*M[k]+.1*g;U[k]=.999*U[k]+.001*g*g;q-=lr*(M[k]/(1-.9**st))/(np.sqrt(U[k]/(1-.999**st))+1e-8)
def games():
    out=[]
    for c in itertools.combinations_with_replacement(range(1,14),4):
        s=sorted(Fr(x) for x in c)
        if solvable(s):out.append(s)
    return out
def sample_states(G,r,n):
    S=[]
    for _ in range(n):
        s=G[int(r.integers(len(G)))]
        for d in range(int(r.integers(0,4))):
            m=moves(s);s=m[int(r.integers(len(m)))][3]
        S.append(s)
    return S
def train_V(G,r,n,steps,seed):
    S=sample_states(G,r,n);X=np.array([feat(s) for s in S]);y=np.array([1. if solvable(s) else 0. for s in S]);m=MLP(X.shape[1],64,seed);m.train(X,lambda i,o:(1/(1+np.exp(-np.clip(o,-30,30)))-y[i]),steps,seed+1);return m,float(y.mean())
def train_P(G,r,n,steps,seed):
    """模仿学习：解题记录里走的那步是正样本，同状态的其它合法步是负样本（softmax 在同一状态内）"""
    X=[];Y=[]
    for _ in range(n):
        s=G[int(r.integers(len(G)))]
        while len(s)>1:
            ms=moves(s);good=[m for m in ms if solvable(m[3])]
            if not good:break
            g=good[int(r.integers(len(good)))];X.append(feat(g[3]));Y.append(1.);b=ms[int(r.integers(len(ms)))];X.append(feat(b[3]));Y.append(1. if solvable(b[3]) else 0.);s=g[3]
    X=np.array(X);Y=np.array(Y);m=MLP(X.shape[1],32,seed);m.train(X,lambda i,o:(1/(1+np.exp(-np.clip(o,-30,30)))-Y[i]),steps,seed+1);return m
def val(V,states):
    if not states:return np.zeros(0)
    return V.fw(np.array([feat(s) for s in states]))[0]
def chain(g,Pn,r,T=1.):
    """一路走到底：每步按 P 分数的 softmax 选一步。返回是否成功、每步后是否还可解"""
    s=g;alive=[];n=0
    while len(s)>1:
        ms=moves(s);q=val(Pn,[m[3] for m in ms])/T;p=np.exp(q-q.max());p/=p.sum();m=ms[int(r.choice(len(ms),p=p))];s=m[3];n+=1;alive.append(solvable(s))
    return s[0]==24,alive,n
def tot(g,Pn,Vn,r,b,k=8,T=1.,noise=.0):
    """BFS：每个状态采 k 个不重复候选，用评估器打分，留最好的 b 个。最后只看第一名是否 24"""
    S=[g];nodes=0;alive=[]
    for t in range(3):
        cand={}
        for s in S:
            ms=moves(s);q=val(Pn,[m[3] for m in ms])/T;p=np.exp(q-q.max());p/=p.sum();idx=r.choice(len(ms),size=min(k,len(ms)),replace=False,p=p)
            for i in idx:cand[tuple(ms[i][3])]=ms[i][3]
        C=list(cand.values());nodes+=len(C);v=val(Vn,C);o=np.argsort(-v)[:b];S=[C[i] for i in o];alive.append(any(solvable(x) for x in S))
    ok=[s[0]==24 for s in S];return ok[0],alive,nodes,any(ok)
def ev(Gt,Pn,Vn,seed,k=100):
    r=np.random.default_rng(seed);res={};cr=[];ca=np.zeros(3);bestk={1:0,5:0,10:0,25:0,100:0};maj=0
    for g in Gt:
        hits=[];vals={}
        for j in range(k):
            ok,al,n=chain(g,Pn,r);hits.append(ok);ca+=np.array(al,float)
        cr.append(np.mean(hits));
        for b_ in bestk:bestk[b_]+=any(hits[:b_])
    res['cot']=float(np.mean(cr));res['alive_cot']=(ca/(k*len(Gt))).tolist();res['best']={str(a):v/len(Gt) for a,v in bestk.items()}
    for b in (1,2,3,5):
        s=0;nn=0;al=np.zeros(3);an=0
        for g in Gt:
            ok,a,nd,ay=tot(g,Pn,Vn,r,b);s+=ok;nn+=nd;al+=np.array(a,float);an+=ay
        res[f'b{b}']=dict(succ=s/len(Gt),nodes=nn/len(Gt),alive=(al/len(Gt)).tolist(),any=an/len(Gt))
    return res
if __name__=='__main__':
    sd=int(sys.argv[1]);t0=time.time();G=games();res={'solvable':len(G)};print('可解局',len(G),round(time.time()-t0),flush=True)
    r=np.random.default_rng(sd);h=[count_paths(g) for g in G];o=np.argsort(np.argsort(-np.array(h)));order=np.argsort(-np.array(h));idx=np.arange(len(G));r2=np.random.default_rng(0)
    Gs=[G[i] for i in order];Tst=Gs[900:1000];Trn=Gs[:900]+Gs[1000:];res['paths_median']=float(np.median(h));res['test_paths']=float(np.median([h[i] for i in order[900:1000]]))
    Vn,base=train_V(Trn,r,8000,6000,sd+1);Pn=train_P(Trn,r,300,2000,sd+2);res['V_base']=base
    St=sample_states(Tst,r,1000);y=np.array([solvable(s) for s in St]);p=val(Vn,St);res['V_acc']=float(((p>0)==y).mean());res['V_pos_rec']=float((p[y]>0).mean());res['V_neg_rec']=float((p[~y]<=0).mean())
    res['hard']=ev(Tst,Pn,Vn,sd+3);print('hard',res['hard'],round(time.time()-t0),flush=True)
    res['easy']=ev(Gs[:100],Pn,Vn,sd+4,k=20);print('easy',{k:(v if not isinstance(v,dict) else v.get('succ')) for k,v in res['easy'].items()},flush=True)
    # 评估器差异：随机评估器、完美评估器；完美生成器
    class RV:
        def fw(s,x):return np.random.default_rng(len(x)+int(1e6*x.sum())%1000).normal(0,1,len(x)),None
    class PV:
        pass
    r3=np.random.default_rng(sd+5);res['abl']={}
    for nm,V,P in (('随机评估器',RV(),Pn),):
        res['abl'][nm]=float(np.mean([tot(g,P,V,r3,5)[0] for g in Tst]))
    def perfect(C):
        return np.array([1. if solvable(s) else 0. for s in C])
    global_val=val
    def valp(V,st):
        return perfect(st) if V=='perfect' else global_val(V,st)
    import builtins;val=valp
    res['abl']['完美评估器']=float(np.mean([tot(g,Pn,'perfect',r3,5)[0] for g in Tst]));res['abl']['完美生成器 + 学到的评估器']=float(np.mean([tot(g,'perfect',Vn,r3,5,T=.05)[0] for g in Tst]));res['abl']['都完美']=float(np.mean([tot(g,'perfect','perfect',r3,5,T=.05)[0] for g in Tst]));val=global_val;print(res['abl'],flush=True)
    res['sec']=time.time()-t0;json.dump(res,open(f'res{sd}.json','w'),ensure_ascii=False)
    if sd==0:np.savez('m_v.npz',**{f'W{i}':w for i,w in enumerate(Vn.W)},**{f'b{i}':b for i,b in enumerate(Vn.b)});np.savez('m_p.npz',**{f'W{i}':w for i,w in enumerate(Pn.W)},**{f'b{i}':b for i,b in enumerate(Pn.b)})
    print('done',round(time.time()-t0),flush=True)
