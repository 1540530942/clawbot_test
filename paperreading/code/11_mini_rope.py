#!/usr/bin/env python3
# 第 11 期迷你 RoPE：python3 11_mini_rope.py 0 1200
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

"""第 11 期玩具实验：四种位置方案（无、学习的绝对位置、正弦相加、RoPE）。
任务 back3：每个位置输出它往前第 3 个词（只和相对距离有关）。
任务 abs5：每个位置都输出“第 5 号格子”的词（只和绝对位置有关）。
都只用长度 16 训练，再测同长度、长度 48 和只看第 32–47 号的“后段”。"""
import sys,json,time,numpy as np
V=12;NUL=0
def batch(r,B,n,task):
    x=r.integers(1,V,(B,n))
    if task=='back3':
        y=np.zeros((B,n),int);y[:,3:]=x[:,:-3];w=np.ones((B,n));w[:,:3]=0
    else:
        y=np.repeat(x[:,5:6],n,1);w=np.ones((B,n))
    return x,y,w
class M(Enc):
    def __init__(s,kind,Lmax=64,seed=0,**k):
        super().__init__(V,Lmax,seed=seed,**k);s.kind=kind;dk=s.d//s.h;s.dk=dk
        s.th=10000.0**(-2*np.arange(dk//2)/dk)
        J=np.zeros((dk,dk))
        for i in range(0,dk,2): J[i,i+1]=1;J[i+1,i]=-1   # 行向量右乘：(x1,x2)->(-x2,x1)
        s.J=T(J)
        p=np.arange(Lmax)[:,None];d=s.d;te=10000.0**(-2*np.arange(d//2)/d);sn=np.zeros((Lmax,d));sn[:,0::2]=np.sin(p*te);sn[:,1::2]=np.cos(p*te);s.sn=sn
    def rot(s,x,n,off):
        a=(np.arange(n)+off)[:,None]*s.th;c=np.repeat(np.cos(a),2,1);t=np.repeat(np.sin(a),2,1)
        return x*c+(x@s.J)*t
    def attn(s,l,x,off=0):
        B,n,_=x.v.shape;h,dk=s.h,s.dk
        sp=lambda t:t.rs(B,n,h,dk).t(0,2,1,3)
        Q,K,W=sp(x@l['q']),sp(x@l['k']),sp(x@l['v'])
        if s.kind=='rope': Q,K=s.rot(Q,n,off),s.rot(K,n,off)
        sc=Q@K.t(0,1,3,2)*(1/np.sqrt(dk));a=sc.softmax();s.last=a.v
        return (a@W).t(0,2,1,3).rs(B,n,h*dk)@l['o']
    def enc(s,x,off=0):
        n=x.shape[1];h=T(np.eye(s.V)[x])@s.E*np.sqrt(s.d)
        if s.kind=='abs': h=h+T(np.eye(s.L)[off:off+n])@s.P
        if s.kind=='sin': h=h+T(s.sn[off:off+n])
        for l in s.layers:
            h=(h+s.attn(l,h,off)).lnorm();h=(h+(((h@l['a'])+l['ab']).relu()@l['b'])+l['bb']).lnorm()
        return h
    def lg(s,x,off=0): return s.enc(x,off)@s.E.t(1,0)
def acc(m,task,n,seed,mask=None,B=150):
    r=np.random.default_rng(seed);x,y,w=batch(r,B,n,task);p=m.lg(x).v.argmax(-1)
    w=w.copy()
    if mask is not None: w=w*mask(n)
    return float(((p==y)*w).sum()/w.sum())
def train(kind,task,seed,steps):
    m=M(kind,seed=seed);r=np.random.default_rng(100+seed);opt=Adam(m.ps);t0=time.time();hist=[]
    for st in range(1,steps+1):
        x,y,w=batch(r,32,16,task);val,p,sd=ce(m.lg(x),y,w);p.backward(sd)
        for q in m.ps:
            if q.g is None:q.g=np.zeros_like(q.v)
        opt.step(min(3e-3,3e-3*st/200)*(1 if st<.7*steps else .3))
        if st%max(1,steps//10)==0: hist.append((st,round(float(val),3)))
    res=dict(curve=hist,same=acc(m,task,16,900+seed))
    res['long']=acc(m,task,48,901+seed)
    res['late']=acc(m,task,48,902+seed,lambda n:np.r_[np.zeros(32),np.ones(n-32)][None,:])
    res['sec']=round(time.time()-t0,1)
    return m,res
if __name__=='__main__':
    sd=int(sys.argv[1]);steps=int(sys.argv[2]) if len(sys.argv)>2 else 1500
    if steps<=0:   # 梯度检查
        m=M('rope',seed=1);x,y,w=batch(np.random.default_rng(0),2,6,'back3')
        for p_ in m.ps:p_.g=None
        v,p,g=ce(m.lg(x),y,w);p.backward(g);P=m.layers[0]['q'];a=P.g[1,2];e=1e-5;P.v[1,2]+=e;v1=ce(m.lg(x),y,w)[0];P.v[1,2]-=2*e;v2=ce(m.lg(x),y,w)[0];P.v[1,2]+=e
        print('grad check',abs(a-(v1-v2)/2/e));sys.exit()
    out={}
    for task in ('back3','abs5'):
        for kind in ('none','abs','sin','rope'):
            m,res=train(kind,task,sd,steps);out[f'{task}/{kind}']=res;print(sd,task,kind,{k:res[k] for k in ('same','long','late','sec')},flush=True)
            json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
            if task=='back3' and kind=='rope' and sd==0:
                np.savez('rope.npz',**{f'p{i}':p_.v for i,p_ in enumerate(m.ps)})
