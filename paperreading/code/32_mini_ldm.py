#!/usr/bin/env python3
# 第 32 期 LDM 第一轮
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

"""第 32 期玩具实验：16×16 手画字形上的潜空间扩散（只用 NumPy，MLP 代替 U-Net 和卷积自编码器）。
先训自编码器压缩到 z，再在 z 上训条件扩散（类别用独热向量拼接，随机丢弃以支持无分类器引导）；对比直接在像素上扩散。这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
G=["..XXXX..,.X....X.,X......X,X......X,X......X,X......X,.X....X.,..XXXX..","...XX...,..XXX...,...XX...,...XX...,...XX...,...XX...,...XX...,.XXXXXX.",".XXXXX..,X.....X.,......X.,.....X..,...XX...,..X.....,.X......,XXXXXXXX",".XXXXX..,......X.,......X.,..XXXX..,......X.,......X.,......X.,.XXXXX..",".....X..,....XX..,...X.X..,..X..X..,.X...X..,XXXXXXXX,.....X..,.....X..","XXXXXXX.,X.......,X.......,XXXXXX..,......X.,......X.,X.....X.,.XXXXX..","..XXXX..,.X......,X.......,XXXXXX..,X.....X.,X.....X.,X.....X.,.XXXXX..","XXXXXXXX,......X.,.....X..,....X...,...X....,...X....,..X.....,..X.....",".XXXXX..,X.....X.,X.....X.,.XXXXX..,X.....X.,X.....X.,X.....X.,.XXXXX..",".XXXXX..,X.....X.,X.....X.,.XXXXXX.,......X.,......X.,.....X..,.XXXX..."]
BM=np.array([[[c=='X' for c in row] for row in g.split(',')] for g in G],float);S=16
def data(r,n,noise=.12):
    y=r.integers(0,10,n);o=np.zeros((n,S,S))
    for i,t in enumerate(y):
        k=r.integers(10,13);ix=(np.arange(k)*8//k);im=BM[t][ix][:,ix];dx,dy=r.integers(0,S-k+1,2);o[i,dy:dy+k,dx:dx+k]=im*r.uniform(.7,1)
    o=o*2-1+r.normal(0,noise,o.shape);return np.clip(o,-1.2,1.2).reshape(n,-1),y
class MLP:
    """通用残差 MLP + Adam。act=relu；输出层很小的初始化"""
    def __init__(s,di,H,do,layers,seed,last=1e-3):
        r=np.random.default_rng(seed);s.W=[r.normal(0,1/np.sqrt(di),(di,H))]+[r.normal(0,1/np.sqrt(H),(H,H)) for _ in range(layers-1)]+[r.normal(0,last,(H,do))];s.b=[np.zeros(H) for _ in range(layers)]+[np.zeros(do)]
        s.P=s.W+s.b;s.M=[np.zeros_like(p) for p in s.P];s.U=[np.zeros_like(p) for p in s.P];s.n=0;s.L=layers
    def fw(s,x):
        hs=[x];h=x
        for i in range(s.L):
            n=np.maximum(h@s.W[i]+s.b[i],0)
            if i>0:n=n+h
            hs.append(n);h=n
        return h@s.W[-1]+s.b[-1],hs
    def grads(s,hs,g):
        """g=dLoss/doutput；返回对输入的梯度和各参数梯度"""
        B=len(g);G=[None]*(2*(s.L+1));G[s.L]=hs[s.L].T@g/B;G[2*s.L+1]=g.mean(0);u=g@s.W[-1].T
        for i in range(s.L-1,-1,-1):
            pre=hs[i];act=(pre@s.W[i]+s.b[i])>0;d=u*act;G[i]=pre.T@d/B;G[s.L+1+i]=d.mean(0);u=d@s.W[i].T+(u if i>0 else 0)
        return u,G
    def update(s,G,lr):
        s.n+=1
        for k,(p,g) in enumerate(zip(s.P,G)):
            s.M[k]=.9*s.M[k]+.1*g;s.U[k]=.999*s.U[k]+.001*g*g;p-=lr*(s.M[k]/(1-.9**s.n))/(np.sqrt(s.U[k]/(1-.999**s.n))+1e-8)
def train_ae(zd,steps,seed):
    r=np.random.default_rng(seed);E=MLP(256,192,zd,2,seed,.05);De=MLP(zd,192,256,2,seed+1,.05)
    for st in range(1,steps+1):
        x,_=data(r,128);z,he=E.fw(x);y,hd=De.fw(z);e=y-x;ud,Gd=De.grads(hd,2*e);ue,Ge=E.grads(he,ud+2*1e-3*z);De.update(Gd,1e-3*(1 if st<.8*steps else .2));E.update(Ge,1e-3*(1 if st<.8*steps else .2))
    x,_=data(np.random.default_rng(seed+9),1000);z=E.fw(x)[0];y=De.fw(z)[0];mse=float(((y-x)**2).mean());return E,De,dict(mse=mse,psnr=float(10*np.log10(4/mse)),zstd=float(z.std()))
def sched(T):
    s=1000/T;b=np.linspace(1e-4*s,.02*s,T);a=1-b;return b,a,np.cumprod(a)
def temb(t,T,d=32):
    f=np.exp(np.linspace(0,np.log(1000),d//2));p=(t/T)[:,None]*f;return np.concatenate([np.sin(p),np.cos(p)],1)
def inp(z,t,T,y):
    return np.c_[z,temb(t,T),np.eye(11)[y]]
def train_dm(net,enc,steps,seed,T=200,scale=1.):
    r=np.random.default_rng(seed);b,a,ab=sched(T);t0=time.time()
    for st in range(1,steps+1):
        x,y=data(r,128);z=(enc.fw(x)[0]/scale) if enc else x;t=r.integers(0,T,128);y=np.where(r.random(128)<.1,10,y);e=r.normal(0,1,z.shape);zt=np.sqrt(ab[t])[:,None]*z+np.sqrt(1-ab[t])[:,None]*e;o,hs=net.fw(inp(zt,t,T,y));_,G=net.grads(hs,2*(o-e));net.update(G,1e-3*(1 if st<.8*steps else .2))
    return (time.time()-t0)/steps
def sample(net,de,n,y,s_,seed,T=200,scale=1.,dim=None):
    r=np.random.default_rng(seed);b,a,ab=sched(T);z=r.normal(0,1,(n,dim));yy=np.full(n,y) if np.isscalar(y) else y;nul=np.full(n,10)
    for t in range(T-1,-1,-1):
        tt=np.full(n,t);ec=net.fw(inp(z,tt,T,yy))[0]
        if s_!=1:eu=net.fw(inp(z,tt,T,nul))[0];ep=eu+s_*(ec-eu)
        else:ep=ec
        z=(z-b[t]/np.sqrt(1-ab[t])*ep)/np.sqrt(a[t])+(np.sqrt(b[t])*r.normal(0,1,z.shape) if t>0 else 0)
    return de.fw(z*scale)[0] if de else z
def train_clf(seed,steps=1500):
    r=np.random.default_rng(seed);c=MLP(256,128,10,2,seed,.05)
    for st in range(steps):
        x,y=data(r,128);z,h=c.fw(x);q=z-z.max(1,keepdims=True);p=np.exp(q);p/=p.sum(1,keepdims=True);p[np.arange(128),y]-=1;_,G=c.grads(h,p);c.update(G,1e-3)
    return c
def feats(c,x):
    z,h=c.fw(x);return h[-1]
def fid(c,x,y):
    a,b=feats(c,x),feats(c,y);m=a.mean(0)-b.mean(0);A=np.cov(a.T)+1e-6*np.eye(a.shape[1]);B=np.cov(b.T)+1e-6*np.eye(a.shape[1]);va,ua=np.linalg.eigh(A);sq=(ua*np.sqrt(np.maximum(va,0)))@ua.T;e=np.linalg.eigvalsh(sq@B@sq);return float(m@m+np.trace(A)+np.trace(B)-2*np.sqrt(np.maximum(e,0)).sum())
def evaluate(net,de,c,ref,dim,scale,seed):
    out={};r=np.random.default_rng(seed);y=r.integers(0,10,800)
    for s_ in (1.,1.5,3.,7.5):
        t=time.time();x=sample(net,de,800,y,s_,seed+1,scale=scale,dim=dim);p=c.fw(x)[0].argmax(1);out[str(s_)]=dict(acc=float((p==y).mean()),fid=fid(c,x,ref),sec=round(time.time()-t,1))
    return out
if __name__=='__main__':
    sd=int(sys.argv[1]);Sd=int(sys.argv[2]) if len(sys.argv)>2 else 6000;Sa=int(sys.argv[3]) if len(sys.argv)>3 else 4000;H=128;out={};t0=time.time()
    c=train_clf(sd+50);rr=np.random.default_rng(sd+60);ref,yr=data(rr,2000);te,yt=data(rr,1000);out['clf']=float((c.fw(te)[0].argmax(1)==yt).mean());out['realfid']=fid(c,te,ref);print(sd,'分类器',out['clf'],'真实 FID',out['realfid'],flush=True)
    net=MLP(256+32+11,H,256,3,sd);sp=train_dm(net,None,Sd,sd+1);out['pixel']=dict(dim=256,sec=sp,flop=int(net.W[0].size+2*H*H+H*256),ev=evaluate(net,None,c,ref,256,1.,sd+2));print(sd,'pixel',out['pixel'],round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'))
    for zd in (64,32,16,8,4):
        E,De,ae=train_ae(zd,Sa,sd+3);x,_=data(np.random.default_rng(1),1000);sc=float(E.fw(x)[0].std());net=MLP(zd+32+11,H,zd,3,sd);sp=train_dm(net,E,Sd,sd+1,scale=sc);r=dict(dim=zd,ae=ae,sec=sp,scale=sc,flop=int(net.W[0].size+2*H*H+H*zd),ev=evaluate(net,De,c,ref,zd,sc,sd+2));out[f'z{zd}']=r;print(sd,zd,r,round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'))
        if zd==16:
            n2=MLP(zd+32+11,H,zd,3,sd);train_dm(n2,E,Sd,sd+1,scale=1.);out['z16_noscale']=dict(ev=evaluate(n2,De,c,ref,zd,1.,sd+2),zstd=sc);print(sd,'noscale',out['z16_noscale'],flush=True)
            if sd==0:np.savez('m.npz',**{f'n_W{i}':w for i,w in enumerate(net.W)},**{f'n_b{i}':w for i,w in enumerate(net.b)},**{f'd_W{i}':w for i,w in enumerate(De.W)},**{f'd_b{i}':w for i,w in enumerate(De.b)},scale=np.array([sc]))
    json.dump(out,open(f'res{sd}.json','w'))
