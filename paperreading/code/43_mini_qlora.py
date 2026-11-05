#!/usr/bin/env python3
# 第 43 期 QLoRA 玩具实验：4 比特量化类型、双重量化前提和 LoRA 微调
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

"""第 43 期玩具实验：QLoRA。先训练一个 MLP（基础模型），把权重分块量化成 4 比特（整数、两种浮点、NF4），只测量化的伤害；
然后冻结量化权重，训练 LoRA 适配到相关的新任务，和 16 比特完整微调比较。还比较只在部分层加 LoRA、不同秩 r、不同分块大小，
并数内存。这是玩具，不代表论文结果。"""
import sys,json,time,math,numpy as np
D=64;K=16;H=256
def ndtri(p):
    lo,hi=-10.,10.
    for _ in range(80):
        m=(lo+hi)/2
        if .5*(1+math.erf(m/math.sqrt(2)))<p:lo=m
        else:hi=m
    return (lo+hi)/2
PAPER=np.array([-1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.0,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.5626170039176941,0.7229568362236023,1.0])
def codebook(name):
    if name=='int4':return np.arange(-7,8)/7.
    if name=='nf4':return PAPER
    if name=='e2m1':
        v=sorted({s*(1+m/2)*2**e for s in(1,-1) for e in range(4) for m in(0,1)});v=[0.]+v;v=sorted(v)[:8]+[0.]+sorted(v)[10:];return np.array(sorted(set(v)))/12.
    if name=='e3m0':
        v=sorted({s*2.**e for s in(1,-1) for e in range(7)});v=[0.]+v;return np.array(sorted(v))[:16]/64.
def quant(w,name,B=64):
    """按块量化：每块除以绝对最大值，映射到最近的码本值，再乘回来。返回解量化后的数组"""
    if name=='bf16':return w.copy()
    cb=codebook(name);f=w.ravel().copy();mids=(cb[1:]+cb[:-1])/2
    for i in range(0,len(f),B):
        b=f[i:i+B];c=np.abs(b).max()+1e-12;idx=np.searchsorted(mids,b/c);f[i:i+B]=cb[idx]*c
    return f.reshape(w.shape)
def task(seed):
    """一个固定的网络规则：输入是 D 维高斯，标签由随机的两层网络产生"""
    r=np.random.default_rng(seed);A=r.normal(0,1,(D,48));B=r.normal(0,1,(48,K));return A,B
def data(rule,n,r):
    A,B=rule;x=r.normal(0,1,(n,D));return x,(np.tanh(x@A/np.sqrt(D)*1.5)@B).argmax(1)
class Net:
    def __init__(s,seed,dropout=0.):
        r=np.random.default_rng(seed);s.W=[r.normal(0,1/np.sqrt(D),(D,H))]+[r.normal(0,1/np.sqrt(H),(H,H)),r.normal(0,1/np.sqrt(H),(H,H)),r.normal(0,1/np.sqrt(H),(H,K))];s.b=[np.zeros(H),np.zeros(H),np.zeros(H),np.zeros(K)];s.A=[None]*4;s.BB=[None]*4;s.alpha=2.
    def fw(s,x):
        hs=[x];h=x
        for i in range(4):
            y=h@s.W[i]+s.b[i]
            if s.A[i] is not None:y=y+s.alpha/s.A[i].shape[1]*(h@s.A[i]@s.BB[i])
            if i<3:y=np.maximum(y,0)
            hs.append(y);h=y
        return h,hs
def xent(lg,y):
    e=lg-lg.max(1,keepdims=True);p=np.exp(e);p/=p.sum(1,keepdims=True);return p
def grads(n,hs,g,base):
    """g=dL/dlogit。返回 base=True 时 W,b 的梯度；A、BB 的梯度"""
    G={};u=g
    for i in range(3,-1,-1):
        pre=hs[i];B=len(pre)
        if i<3:u=u*(hs[i+1]>0)
        if base:G[f'W{i}']=pre.T@u/B;G[f'b{i}']=u.mean(0)
        if n.A[i] is not None:
            sc=n.alpha/n.A[i].shape[1];G[f'A{i}']=sc*pre.T@(u@n.BB[i].T)/B;G[f'B{i}']=sc*(pre@n.A[i]).T@u/B
        J=n.W[i]
        v=u@J.T
        if n.A[i] is not None:v=v+(n.alpha/n.A[i].shape[1])*(u@n.BB[i].T@n.A[i].T)
        u=v
    return G
def params(n,G):
    o={}
    for k in G:
        o[k]=(n.W[int(k[1])] if k[0]=='W' else n.b[int(k[1])] if k[0]=='b' else n.A[int(k[1])] if k[0]=='A' else n.BB[int(k[1])])
    return o
def adam(n,X,y,steps,base,seed,lr,bs=128):
    r=np.random.default_rng(seed);M={};U={}
    for st in range(1,steps+1):
        i=r.integers(0,len(X),bs);lg,hs=n.fw(X[i]);p=xent(lg,y[i]);p2=p.copy();p2[np.arange(bs),y[i]]-=1;G=grads(n,hs,p2,base);P=params(n,G);L=lr*(1 if st<.8*steps else .2)
        for k,g in G.items():
            M[k]=M.get(k,0)*.9+.1*g;U[k]=U.get(k,0)*.999+.001*g*g;P[k]-=L*(M[k]/(1-.9**st))/(np.sqrt(U[k]/(1-.999**st))+1e-8)
def acc(n,X,y):
    return float((n.fw(X)[0].argmax(1)==y).mean())
def lora(n,r_,seed,layers=(0,1,2,3)):
    r=np.random.default_rng(seed)
    for i in layers:
        a=n.W[i].shape;n.A[i]=r.normal(0,1/np.sqrt(a[0]),(a[0],r_));n.BB[i]=np.zeros((r_,a[1]))
def copy(n):
    m=Net(0);m.W=[w.copy() for w in n.W];m.b=[b.copy() for b in n.b];return m
def quantize(n,name,B=64):
    m=copy(n);m.W=[quant(w,name,B) for w in m.W];return m
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 4000;F=int(sys.argv[3]) if len(sys.argv)>3 else 1200;t0=time.time();res={};r=np.random.default_rng(sd)
    off=.9677083;v=np.array(sorted([ndtri(p) for p in np.linspace(off,.5,9)[:-1]]+[-ndtri(p) for p in np.linspace(off,.5,8)[:-1]]+[0.]));res['nf4_refit_err']=float(abs(v/abs(v).max()-PAPER).max())
    q4=np.array([(ndtri(i/17)+ndtri((i+1)/17))/2 for i in range(1,16)]);res['nf4_eq4']=np.round(q4/abs(q4).max(),4).tolist()
    # 各类型把标准正态的 64 个一块量化后，每个码用到的比例（信息量）
    g=np.random.default_rng(sd+9).normal(0,1,(4000,64));res['occ']={}
    for k in ('int4','e2m1','e3m0','nf4'):
        cb=codebook(k);mids=(cb[1:]+cb[:-1])/2;idx=np.searchsorted(mids,(g/np.abs(g).max(1,keepdims=True)).ravel());c=np.bincount(idx,minlength=len(cb))/idx.size;res['occ'][k]=dict(p=np.round(c,4).tolist(),H=float(-(c[c>0]*np.log2(c[c>0])).sum()),mse=float((((g/np.abs(g).max(1,keepdims=True)).ravel()-cb[idx])**2).mean()))
    # 基础任务
    Ra=task(100);Xa,ya=data(Ra,40000,r);Xt,yt=data(Ra,5000,r);n=Net(sd);adam(n,Xa,ya,S,True,sd+1,1e-3);res['base_acc']=acc(n,Xt,yt);print('base',res['base_acc'],round(time.time()-t0),flush=True)
    ws=np.concatenate([w.ravel() for w in n.W[1:3]]);m,s_=ws.mean(),ws.std();res['normal']=dict(mean=float(m),std=float(s_),kurt=float((((ws-m)/s_)**4).mean()-3),mx=float(np.abs(ws).max()/s_),hist=np.histogram(ws/s_,bins=np.linspace(-4,4,33))[0].tolist())
    np.savez('m_w.npz',w=np.round(ws[:4096],5))
    # 只量化的伤害
    res['inf']={};kinds=('int4','e2m1','e3m0','nf4')
    for k in kinds:
        q=quantize(n,k);res['inf'][k]=dict(acc=acc(q,Xt,yt),mse=float(np.mean([((a-b)**2).mean()/(a**2).mean() for a,b in zip(n.W,q.W)])));
    res['inf_b']={str(B):acc(quantize(n,'nf4',B),Xt,yt) for B in (8,16,64,256,1024,4096,10**9)};res['inf_b_int']={str(B):acc(quantize(n,'int4',B),Xt,yt) for B in (8,16,64,256,1024,4096,10**9)};print(res['inf'],res['inf_b'],flush=True)
    # 新任务：规则相近
    A,B=Ra;rb=np.random.default_rng(55);Rb=(A+.8*rb.normal(0,1,A.shape),B+.8*rb.normal(0,1,B.shape));Xb,yb=data(Rb,3000,r);Xe,ye=data(Rb,5000,r);res['before']=acc(n,Xe,ye)
    fin={}
    def run(tag,q,r_=16,layers=(0,1,2,3),full=False,bs=128):
        m=quantize(n,q);
        if full:adam(m,Xb,yb,F,True,sd+2,3e-4)
        else:lora(m,r_,sd+3,layers);adam(m,Xb,yb,F,False,sd+2,1e-3)
        fin[tag]=acc(m,Xe,ye);print(sd,tag,fin[tag],round(time.time()-t0),flush=True);return m
    run('full16','bf16',full=True);run('lora16','bf16')
    for k in kinds:run('lora_'+k,k)
    for r_ in (2,4,8,16,32,64):run(f'r{r_}','nf4',r_)
    run('partial','nf4',16,(1,));run('last','nf4',16,(3,));run('first2','nf4',16,(0,1))
    res['fin']=fin
    # 内存（比特/参数）
    res['sec']=time.time()-t0;json.dump(res,open(f'res{sd}.json','w'),ensure_ascii=False);print('done',round(time.time()-t0),flush=True)
