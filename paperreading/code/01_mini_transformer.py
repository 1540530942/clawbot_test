#!/usr/bin/env python3
"""第 1 期配套：只用 NumPy 从零训练迷你 Transformer，把 6 个数字倒过来。
运行：python3 01_mini_transformer.py 1200（2 核 CPU 约 1 分钟）。
"""
import sys
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
V=12; BOS=10; PAD=11; L=6
def pe(n,d):
    p=np.zeros((n,d)); pos=np.arange(n)[:,None]; i=np.arange(0,d,2)
    p[:,0::2]=np.sin(pos/10000**(i/d)); p[:,1::2]=np.cos(pos/10000**(i/d)); return p
class M:
    def __init__(s,d,h,ff,N,seed,use_pe=True,use_mask=True):
        r=np.random.default_rng(seed); s.d,s.h,s.ff,s.N,s.use_pe,s.use_mask=d,h,ff,N,use_pe,use_mask; s.ps=[]
        s.E=s.par(r.normal(0,d**-.5,(V,d)))
        def lin(a,b): return s.par(r.normal(0,1/np.sqrt(a),(a,b)))
        def mha(): return dict(q=lin(d,d),k=lin(d,d),v=lin(d,d),o=lin(d,d))
        def ffn(): return dict(a=lin(d,ff),ab=s.par(np.zeros(ff)),b=lin(ff,d),bb=s.par(np.zeros(d)))
        s.enc=[dict(a=mha(),f=ffn()) for _ in range(N)]; s.dec=[dict(a=mha(),c=mha(),f=ffn()) for _ in range(N)]
        s.P=pe(L+2,d); s.cross=None
    def par(s,v): t=T(v); s.ps.append(t); return t
    def embed(s,x):
        o=np.eye(V)[x]; e=T(o)@s.E*np.sqrt(s.d)
        return e+s.P[:x.shape[1]][None] if s.use_pe else e
    def attn(s,m,q,k,v,mask=None):
        B,nq,_=q.v.shape; nk=k.v.shape[1]; h,dk=s.h,s.d//s.h
        sp=lambda t,n:t.rs(B,n,h,dk).t(0,2,1,3)
        Q,K,W=sp(q@m['q'],nq),sp(k@m['k'],nk),sp(v@m['v'],nk)
        sc=Q@K.t(0,1,3,2)*(1/np.sqrt(dk))
        if mask is not None:
            sc=T(np.where(mask,sc.v,-1e9),(sc,)); sc.bk=lambda g,mask=mask:(g*mask,)
        a=sc.softmax(); s.last=a.v
        return (a@W).t(0,2,1,3).rs(B,nq,h*dk)@m['o']
    def ffn(s,f,x): return (((x@f['a'])+f['ab']).relu()@f['b'])+f['bb']
    def forward(s,src,tgt):
        x=s.embed(src)
        for l in s.enc:
            x=(x+s.attn(l['a'],x,x,x)).lnorm(); x=(x+s.ffn(l['f'],x)).lnorm()
        n=tgt.shape[1]; mask=np.tril(np.ones((n,n),bool)) if s.use_mask else None
        y=s.embed(tgt)
        for l in s.dec:
            y=(y+s.attn(l['a'],y,y,y,mask)).lnorm(); s.self_attn=s.last
            y=(y+s.attn(l['c'],y,x,x)).lnorm(); s.cross=s.last
            y=(y+s.ffn(l['f'],y)).lnorm()
        return y@s.E.t(1,0)   # 输出层和词向量共享
def batch(r,B):
    a=r.integers(0,10,(B,L)); b=a[:,::-1]
    return a,np.c_[np.full(B,BOS),b[:,:-1]],b
def loss(lg,y,eps=.1):
    p=lg.softmax(); t=np.full((*y.shape,V),eps/(V-1)); np.put_along_axis(t,y[...,None],1-eps,-1)
    lp=np.log(p.v+1e-12); val=-(t*lp).sum(-1).mean()
    seed=-(t/(p.v+1e-12))/np.prod(y.shape); return val,p,seed
def greedy(m,a):
    y=np.full((len(a),1),BOS)
    for _ in range(L):
        lg=m.forward(a,y); y=np.c_[y,lg.v[:,-1].argmax(-1)]
    return y[:,1:]
def evaluate(m,seed=999,B=2000):
    r=np.random.default_rng(seed); a,ti,b=batch(r,B)
    tf=(m.forward(a,ti).v.argmax(-1)==b).mean()
    g=greedy(m,a); return float(tf),float((g==b).mean()),float((g==b).all(1).mean())
def train(steps,d=32,h=4,ff=64,N=1,seed=0,warm=200,use_pe=True,use_mask=True,B=64,log=None):
    m=M(d,h,ff,N,seed,use_pe,use_mask); o=Adam(m.ps); r=np.random.default_rng(seed+1000); hist=[]
    for st in range(1,steps+1):
        a,ti,b=batch(r,B); lg=m.forward(a,ti); v,p,sd=loss(lg,b); p.backward(sd)
        lr=d**-.5*min(st**-.5,st*warm**-1.5)
        o.step(lr)
        if st%100==0 or st==1: hist.append((st,round(v,4)))
    return m,hist
def gradcheck():
    m=M(8,2,8,1,0); r=np.random.default_rng(0); a,ti,b=batch(r,4); lg=m.forward(a,ti); v,p,sd=loss(lg,b); p.backward(sd)
    worst=0
    for P in m.ps[:6]:
        idx=tuple(r.integers(0,s) for s in P.v.shape); g=P.g[idx]; e=1e-5; o=P.v[idx]
        P.v[idx]=o+e; a1=loss(m.forward(a,ti),b)[0]; P.v[idx]=o-e; a2=loss(m.forward(a,ti),b)[0]; P.v[idx]=o
        worst=max(worst,abs((a1-a2)/(2*e)-g))
    return worst
if __name__=='__main__':
    print('梯度检验最大误差',gradcheck())
    m,h=train(int(sys.argv[1]) if len(sys.argv)>1 else 1200)
    print('最后损失',h[-1],'（喂答案每位准确率, 自己写每位准确率, 整题全对）',evaluate(m))
    print('314159 ->',greedy(m,np.array([[3,1,4,1,5,9]]))[0].tolist())
