#!/usr/bin/env python3
# 第 9 期迷你 ViT：python3 09_mini_vit.py 0 600
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

"""迷你 ViT：合成线条图形分类。比较 ViT、去掉位置的 ViT、小卷积网，数据量变化时谁更好。"""
import sys,json,numpy as np
H=16;P=4;NP=(H//P)**2;CL=['横线','竖线','方框','十字','对角线','小点']
def draw(r,c):
    a=np.zeros((H,H));s=r.integers(5,10);x=r.integers(0,H-s+1);y=r.integers(0,H-s+1)
    if c==0:a[y+s//2,x:x+s]=1
    elif c==1:a[y:y+s,x+s//2]=1
    elif c==2:a[y,x:x+s]=1;a[y+s-1,x:x+s]=1;a[y:y+s,x]=1;a[y:y+s,x+s-1]=1
    elif c==3:a[y+s//2,x:x+s]=1;a[y:y+s,x+s//2]=1
    elif c==4:
        for i in range(s):a[y+i,x+(i if r.random()<2 else 0)]=1
    else:a[y+s//2:y+s//2+2,x+s//2:x+s//2+2]=1
    return np.clip(a+r.normal(0,.12,a.shape),0,1)
def data(r,n):
    y=r.integers(0,6,n);return np.stack([draw(r,c) for c in y]),y
def patches(X):
    B=len(X);return X.reshape(B,H//P,P,H//P,P).transpose(0,1,3,2,4).reshape(B,NP,P*P)
class ViT:
    def __init__(s,d=32,h=4,ff=64,N=2,seed=0,pos=True):
        r=np.random.default_rng(seed);s.d,s.h,s.N,s.pos=d,h,N,pos;s.ps=[];lin=lambda a,b:s.par(r.normal(0,1/np.sqrt(a),(a,b)))
        s.E=lin(P*P,d);s.C=s.par(r.normal(0,.1,(1,d)));s.Pz=s.par(r.normal(0,.1,(NP+1,d)))
        s.L=[dict(q=lin(d,d),k=lin(d,d),v=lin(d,d),o=lin(d,d),a=lin(d,ff),ab=s.par(np.zeros(ff)),b=lin(ff,d),bb=s.par(np.zeros(d))) for _ in range(N)]
        s.Hd=lin(d,6)
    def par(s,v):t=T(v);s.ps.append(t);return t
    def attn(s,l,x):
        B,n,_=x.v.shape;h,dk=s.h,s.d//s.h;sp=lambda t:t.rs(B,n,h,dk).t(0,2,1,3);Q,K,W=sp(x@l['q']),sp(x@l['k']),sp(x@l['v'])
        a=(Q@K.t(0,1,3,2)*(1/np.sqrt(dk))).softmax();s.last=a.v;return (a@W).t(0,2,1,3).rs(B,n,h*dk)@l['o']
    def __call__(s,X):
        B=len(X);p=T(patches(X))@s.E
        cls=T(np.ones((B,1,1)))@s.C   # (B,1,d)
        z=T(np.eye(NP+1)[1:][None].repeat(B,0))
        # 拼接：用矩阵把 CLS 和图块放进 NP+1 个位置
        A=np.zeros((1,NP+1));A[0,0]=1
        x=T(np.tile(np.eye(NP+1)[:,1:],(B,1,1)))@p+T(np.tile(A.T,(B,1,1)))@cls
        if s.pos:x=x+s.Pz
        s.maps=[]
        for l in s.L:
            x=x+s.attn(l,x.lnorm());s.maps.append(s.last);g=x.lnorm();x=x+(((g@l['a'])+l['ab']).relu()@l['b'])+l['bb']
        pick=np.zeros((B,NP+1,1));pick[:,0,0]=1
        return (x.lnorm().t(0,2,1)@T(pick)).rs(B,s.d)@s.Hd
def take(x,idx):
    """按下标取第二维（卷积窗口展开），反向时把梯度放回去。"""
    r=T(x.v[:,idx],(x,))
    def bk(g):
        o=np.zeros_like(x.v);np.add.at(o,(slice(None),idx),g);return (o,)
    r.bk=bk;return r
class CNN:
    """两层 5×5 卷积加全局平均池化，用矩阵展开实现，参数量和 ViT 同级。"""
    def __init__(s,c1=8,c2=48,seed=0):
        r=np.random.default_rng(seed);s.ps=[];s.c1,s.c2=c1,c2;s.k=5
        s.W1=s.par(r.normal(0,1/5,(25,c1)));s.W2=s.par(r.normal(0,1/np.sqrt(25*c1),(25*c1,c2)));s.Hd=s.par(r.normal(0,1/np.sqrt(c2),(c2,6)))
        s.n1=H-4;s.n2=s.n1-4
        s.U1=s.unf(H);s.U2=s.unf(s.n1)
    def par(s,v):t=T(v);s.ps.append(t);return t
    def unf(s,n):
        return np.array([(y+j)*n+x+i for y in range(n-4) for x in range(n-4) for j in range(5) for i in range(5)])
    def __call__(s,X):
        B=len(X);a=take(T(X.reshape(B,H*H,1)),s.U1);a=(a.rs(B,s.n1**2,25)@s.W1).relu()
        b=take(a,s.U2);b=(b.rs(B,s.n2**2,25*s.c1)@s.W2).relu()
        g=b.t(0,2,1)@T(np.ones((B,s.n2**2,1))/s.n2**2)
        return g.rs(B,s.c2)@s.Hd
def run(m,Xt,yt,Xe,ye,steps,seed):
    o=Adam(m.ps,b2=.999);r=np.random.default_rng(seed+11)
    for st in range(1,steps+1):
        i=r.integers(0,len(Xt),48);lg=m(Xt[i]);v,p,sd=ce(lg,yt[i]);p.backward(sd);o.step(min(st/50,1)*2e-3*(1-st/(steps+1)))
    return float((m(Xe).v.argmax(-1)==ye).mean()),float((m(Xt[:min(len(Xt),500)]).v.argmax(-1)==yt[:500]).mean())
if __name__=='__main__':
    seed=int(sys.argv[1]);steps=int(sys.argv[2]);r=np.random.default_rng(seed+1);Xe,ye=data(np.random.default_rng(99),1000);out=[]
    for n in (100,1000,10000):
        Xt,yt=data(r,n)
        for name,mk in (('vit',lambda:ViT(seed=seed)),('nopos',lambda:ViT(seed=seed,pos=False)),('cnn',lambda:CNN(seed=seed))):
            m=mk();t,tr=run(m,Xt,yt,Xe,ye,steps,seed);out.append(dict(n=n,model=name,test=t,train=tr,params=sum(p.v.size for p in m.ps)));print(n,name,t,tr,out[-1]['params'],flush=True)
            if seed==0 and n==10000 and name=='vit':np.savez('vit.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)})
        json.dump(out,open(f'res{seed}.json','w'))
