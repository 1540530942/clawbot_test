#!/usr/bin/env python3
"""第 5 期迷你 T5（NumPy），完整可运行：把自动微分、Transformer 辅助函数和实验拼成一个文件。
用法：python3 05_mini_t5.py 0 2000（约一小时，单核 CPU）。"""
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
            s.m[i]=s.b1*s.m[i]+(1-s.b1)*p.g; s.u[i]=s.b2*s.u[i]+(1-s.b2)*p.g*p.g
            p.v-=lr*(s.m[i]/(1-s.b1**s.n))/(np.sqrt(s.u[i]/(1-s.b2**s.n))+s.eps); p.g=None

import sys,json
def ce(lg,y,w=None):
    """交叉熵。w 是每个位置的权重（如只算被遮的词）。返回 损失值, 概率节点, 反向种子"""
    p=lg.softmax(); idx=np.indices(y.shape); py=p.v[(*idx,y)]
    if w is None: w=np.ones(y.shape)
    W=w.sum()+1e-9; val=-(w*np.log(py+1e-12)).sum()/W
    sd=np.zeros(p.v.shape); sd[(*idx,y)]=-w/(py+1e-12)/W
    return val,p,sd

DIG=[str(i) for i in range(10)]
vocab=['<p>','<s>','<e>']+DIG+['倒序:','排序:','奇偶:','平均:','奇','偶','.']
ID={w:i for i,w in enumerate(vocab)};V=len(vocab);LS=7;LT=7;PAD=0;BOS=1;EOS=2
def ids(ws,n): return [ID[w] for w in ws]+[PAD]*(n-len(ws))
def make(task,d):
    """d 是数字列表，返回 (输入词, 目标词)"""
    s=[str(x) for x in d]
    if task=='倒序:':return [task]+s,s[::-1]
    if task=='排序:':return [task]+s,[str(x) for x in sorted(d)]
    if task=='奇偶:':return [task]+s,['奇' if sum(d)%2 else '偶']
    a=sum(d[:2])/2;return [task]+s[:2],list(f'{a:.1f}')   # 回归写成文本：3 7 -> 5.0
TASKS=['倒序:','排序:','奇偶:','平均:']
def pool(r,task,n):
    if task=='平均:':
        all_=[(a,b) for a in range(10) for b in range(10)];idx=r.permutation(100)[:n];return [list(all_[i]) for i in idx]
    return [list(r.integers(0,10,5)) for _ in range(n)]
class ED:
    def __init__(s,d=48,h=4,ff=96,N=2,seed=0):
        r=np.random.default_rng(seed);s.d,s.h,s.N=d,h,N;s.ps=[]
        s.E=s.par(r.normal(0,d**-.5,(V,d)));s.Pe=s.par(r.normal(0,d**-.5,(LS,d)));s.Pd=s.par(r.normal(0,d**-.5,(LT,d)))
        lin=lambda a,b:s.par(r.normal(0,1/np.sqrt(a),(a,b)))
        mh=lambda:dict(q=lin(d,d),k=lin(d,d),v=lin(d,d),o=lin(d,d));fw=lambda:dict(a=lin(d,ff),ab=s.par(np.zeros(ff)),b=lin(ff,d),bb=s.par(np.zeros(d)))
        s.enc=[dict(a=mh(),f=fw()) for _ in range(N)];s.dec=[dict(a=mh(),c=mh(),f=fw()) for _ in range(N)]
    def par(s,v):t=T(v);s.ps.append(t);return t
    def attn(s,m,q,k,mask=None):
        B,nq,_=q.v.shape;nk=k.v.shape[1];h,dk=s.h,s.d//s.h
        sp=lambda t,n:t.rs(B,n,h,dk).t(0,2,1,3);Q,K,W=sp(q@m['q'],nq),sp(k@m['k'],nk),sp(k@m['v'],nk)
        sc=Q@K.t(0,1,3,2)*(1/np.sqrt(dk))
        if mask is not None:
            o=sc;sc=T(np.where(mask,o.v,-1e9),(o,));sc.bk=lambda g,mask=mask:(g*mask,)
        return (sc.softmax()@W).t(0,2,1,3).rs(B,nq,h*dk)@m['o']
    def ffn(s,f,x):return (((x@f['a'])+f['ab']).relu()@f['b'])+f['bb']
    def emb(s,x,P):
        n=x.shape[1];return T(np.eye(V)[x])@s.E*np.sqrt(s.d)+T(np.eye(P.v.shape[0])[:n])@P
    def encode(s,x):
        h=s.emb(x,s.Pe)
        for l in s.enc:
            h=h+s.attn(l['a'],h.lnorm(),h.lnorm());h=h+s.ffn(l['f'],h.lnorm())
        return h.lnorm()
    def decode(s,y,mem):
        n=y.shape[1];mask=np.tril(np.ones((n,n),bool));h=s.emb(y,s.Pd)
        for l in s.dec:
            h=h+s.attn(l['a'],h.lnorm(),h.lnorm(),mask);h=h+s.attn(l['c'],h.lnorm(),mem);h=h+s.ffn(l['f'],h.lnorm())
        return h.lnorm()@s.E.t(1,0)
def to_arrays(items):
    X=np.array([ids(a,LS) for a,b in items]);Y=np.array([ids(['<s>']+b+['<e>'],LT+1) for a,b in items]);return X,Y
def greedy(m,X):
    mem=m.encode(X);y=np.full((len(X),1),BOS);done=np.zeros(len(X),bool)
    for _ in range(LT-1):
        nx=m.decode(y,mem).v[:,-1];nx[:,PAD]=-1e9;nx[:,BOS]=-1e9;k=nx.argmax(-1);y=np.c_[y,k]
    return y[:,1:]
def text(row):
    o=[]
    for t in row:
        if t==EOS:break
        o.append(vocab[t])
    return o
def evaluate(m,seed=2024,avg_test=None):
    r=np.random.default_rng(seed);res={}
    for t in TASKS:
        ds=[list(x) for x in (avg_test if t=='平均:' and avg_test else pool(r,t,100 if t=='平均:' else 600))];items=[make(t,d) for d in ds];X,Y=to_arrays(items)
        G=greedy(m,X);ok=[text(G[i])==items[i][1] for i in range(len(items))];res[t]=float(np.mean(ok))
        if t=='平均:':
            er=[]
            for i,d in enumerate(ds):
                try:er.append(abs(float(''.join(text(G[i])))-sum(d[:2])/2))
                except Exception:er.append(np.nan)
            res['平均误差']=float(np.nanmean(er));res['无效']=float(np.mean(np.isnan(er)))
    return res
SIZES={'倒序:':20000,'排序:':20000,'奇偶:':300,'平均:':60}
def train(seed,mix,steps=3000,every=500,B=64,K=None,T_=None):
    r=np.random.default_rng(seed+5);P={t:pool(r,t,SIZES[t]) for t in TASKS};held=[[a,b] for a in range(10) for b in range(10) if [a,b] not in [list(x) for x in P['平均:']]]
    m=ED(seed=seed);o=Adam(m.ps,b2=.999);rr=np.random.default_rng(seed+9)
    sz=np.array([SIZES[t] for t in TASKS],float)
    if mix=='equal':w=np.ones(4)/4
    elif mix=='prop':w=sz/sz.sum()
    elif mix=='limit':w=np.minimum(sz,K);w=w/w.sum()
    elif mix=='temp':w=sz/sz.sum();w=w**(1/T_);w=w/w.sum()
    hist=[]
    for st in range(1,steps+1):
        ts=rr.choice(4,B,p=w);items=[make(TASKS[i],P[TASKS[i]][rr.integers(len(P[TASKS[i]]))]) for i in ts]
        X,Y=to_arrays(items);tgt=Y[:,1:];lg=m.decode(Y[:,:-1],m.encode(X));val,p,sd=ce(lg,tgt,(tgt!=PAD)*1.);p.backward(sd)
        o.step(min(st/200,1)*3e-3*(1-st/(steps+1)))
        if st%every==0:hist.append((st,evaluate(m,avg_test=held)))
    return m,w.tolist(),hist
if __name__=='__main__':
    seed=int(sys.argv[1]);steps=int(sys.argv[2]);out=[]
    for name,kw in [('equal',dict(mix='equal')),('prop',dict(mix='prop')),('limit',dict(mix='limit',K=2000)),('temp',dict(mix='temp',T_=2))]:
        m,w,h=train(seed,steps=steps,**kw);out.append(dict(name=name,w=w,hist=h,final=h[-1][1]));print(seed,name,[round(x,4) for x in w],h[-1][1],flush=True)
        if seed==0 and name=='limit':np.savez('t5.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)})
        json.dump(out,open(f'res{seed}.json','w'),ensure_ascii=False)
