#!/usr/bin/env python3
# 第 39 期 ACT 玩具实验：动作分块、时间集成、CVAE、L1 和 L2
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

"""第 39 期玩具实验：动作分块。一条弯曲的窄通道，执行有误差（便宜的机械臂），示范里有停顿（非马尔可夫），
“人类”数据还在一处有两种绕法。比较分块长度 k、时间集成、CVAE、L1/L2。MLP 代替 Transformer，没有图像。这是玩具。"""
import sys,json,time,numpy as np
T=200;W=.04;Z=8
def path(u):
    """通道中心线，u∈[0,1]"""
    return np.stack([2*u-1,.45*np.sin(5*u)*np.cos(2.2*u)],-1)
def bump(u,side):
    """第 2 段的绕法：u 在 .55–.75 之间向上或向下绕"""
    v=np.clip((u-.55)/.2,0,1);return side*.12*np.sin(np.pi*v)
def demo(r,human):
    """返回目标位置序列。人类有速度变化、停顿、抖动和两种绕法"""
    n=int(r.integers(150,190));s=np.ones(n)
    if human:
        s=np.maximum(.3,1+np.cumsum(r.normal(0,.05,n)));s=np.minimum(s,1.7)
    for _ in range(int(r.integers(1,4))):
        i=int(r.integers(10,n-20));s[i:i+int(r.integers(6,14))]=0
    u=np.cumsum(s);u=u/u[-1];c=path(u)
    if human:
        c=c+np.c_[np.zeros(n),bump(u,1. if r.random()<.5 else -1.)];c=c+np.cumsum(r.normal(0,.0008,c.shape),0)
    return c
def inside(p,human):
    """到中心线的距离。人类示范的绕法也算在通道内"""
    u=np.linspace(0,1,400);c=path(u);d=np.linalg.norm(p[None]-c,axis=1).min()
    if human:
        for sd in (1,-1):
            c2=c+np.c_[np.zeros(400),bump(u,sd)];d=min(d,np.linalg.norm(p[None]-c2,axis=1).min())
    return d
class MLP:
    def __init__(s,di,H,do,L,seed,last=1e-2):
        r=np.random.default_rng(seed);s.W=[r.normal(0,1/np.sqrt(di),(di,H))]+[r.normal(0,1/np.sqrt(H),(H,H)) for _ in range(L-1)]+[r.normal(0,last,(H,do))];s.b=[np.zeros(H) for _ in range(L)]+[np.zeros(do)];s.P=s.W+s.b;s.M=[np.zeros_like(p) for p in s.P];s.U=[np.zeros_like(p) for p in s.P];s.n=0;s.L=L
    def fw(s,x):
        hs=[x];h=x
        for i in range(s.L):
            n=np.maximum(h@s.W[i]+s.b[i],0)
            if i>0:n=n+h
            hs.append(n);h=n
        return h@s.W[-1]+s.b[-1],hs
    def grads(s,hs,g):
        B=len(g);G=[None]*(2*(s.L+1));G[s.L]=hs[s.L].T@g/B;G[2*s.L+1]=g.mean(0);u=g@s.W[-1].T
        for i in range(s.L-1,-1,-1):
            pre=hs[i];act=(pre@s.W[i]+s.b[i])>0;d=u*act;G[i]=pre.T@d/B;G[s.L+1+i]=d.mean(0);u=d@s.W[i].T+(u if i>0 else 0)
        return u,G
    def upd(s,G,lr):
        s.n+=1
        for k,(p,g) in enumerate(zip(s.P,G)):
            s.M[k]=.9*s.M[k]+.1*g;s.U[k]=.999*s.U[k]+.001*g*g;p-=lr*(s.M[k]/(1-.9**s.n))/(np.sqrt(s.U[k]/(1-.999**s.n))+1e-8)
def data(D,k):
    O=[];A=[]
    for c in D:
        for t in range(len(c)-1):
            idx=np.minimum(np.arange(t+1,t+1+k),len(c)-1);O.append(c[t]);A.append(c[idx].reshape(-1))
    return np.array(O),np.array(A)
class Pol:
    def __init__(s,k,seed,cvae,loss='l1',H=128):
        s.k,s.cvae,s.loss=k,cvae,loss;s.dec=MLP(2+(Z if cvae else 0),H,2*k,3,seed)
        if cvae:s.enc=MLP(2+2*k,H,2*Z,2,seed+1,.01)
    def train(s,O,A,steps,seed,beta=10.):
        """论文口径：L1 按维度平均 + beta*KL。这里 L1 按维度求和，所以 KL 乘 2k，两者只差一个总比例，Adam 不在乎"""
        r=np.random.default_rng(seed);beta=beta*2*s.k
        for st in range(1,steps+1):
            i=r.integers(0,len(O),128);o,a=O[i],A[i];lr=1e-3*(1 if st<.8*steps else .2)
            if s.cvae:
                e,he=s.enc.fw(np.c_[o,a]);mu,lv=e[:,:Z],np.clip(e[:,Z:],-8,3);eps=r.normal(0,1,mu.shape);z=mu+np.exp(.5*lv)*eps;y,hd=s.dec.fw(np.c_[o,z]);g=(np.sign(y-a) if s.loss=='l1' else 2*(y-a));u,Gd=s.dec.grads(hd,g);uz=u[:,2:];ge=np.c_[uz+beta*mu,uz*np.exp(.5*lv)*eps*.5+beta*.5*(np.exp(lv)-1)];_,Ge=s.enc.grads(he,ge);s.dec.upd(Gd,lr);s.enc.upd(Ge,lr)
            else:
                y,hd=s.dec.fw(o);g=(np.sign(y-a) if s.loss=='l1' else 2*(y-a));_,Gd=s.dec.grads(hd,g);s.dec.upd(Gd,lr)
    def predict(s,o):
        x=np.c_[o,np.zeros((len(o),Z))] if s.cvae else o;return s.dec.fw(x)[0].reshape(len(o),s.k,2)
def rollout(p,human,te,seed,n=60,m=.05,noise=.006,gain=.6):
    """执行：目标位置被低增益跟踪，加噪声（便宜硬件）。te=True 时每步预测并按指数权重平均同一时刻的预测"""
    r=np.random.default_rng(seed);x=np.tile(path(np.array([0.]))[0],(n,1));x=x+r.normal(0,.005,x.shape);B=[[[] for _ in range(n)] for _ in range(T+p.k+2)];Q=None;tr=[x.copy()];off=np.zeros(n,bool);prog=np.zeros(n)
    u=np.linspace(0,1,400);c=path(u)
    for t in range(T):
        if te:
            q=p.predict(x)
            for i in range(n):
                for j in range(p.k):B[t+j][i].append(q[i,j])
            tg=np.zeros((n,2))
            for i in range(n):
                A=np.array(B[t][i]);w=np.exp(-m*np.arange(len(A)));tg[i]=(A*w[:,None]).sum(0)/w.sum()
        else:
            if t%p.k==0:Q=p.predict(x);t0=t
            tg=Q[:,t-t0]
        x=x+gain*(tg-x)+r.normal(0,noise,x.shape);tr.append(x.copy())
        for i in range(n):
            if not off[i]:
                d=np.linalg.norm(x[i][None]-c,axis=1);j=int(d.argmin());prog[i]=max(prog[i],u[j])
                if inside(x[i],human)>W*2.2:off[i]=True
    tr=np.stack(tr,1);succ=(prog>.97)&~off;stuck=(np.linalg.norm(tr[:,-1]-tr[:,-40],axis=1)<.02)&(prog<.97)&~off;return dict(succ=float(succ.mean()),off=float(off.mean()),stuck=float(stuck.mean()),prog=float(prog.mean())),tr
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 4000;out={};t0=time.time();r=np.random.default_rng(sd);Ds=[demo(r,False) for _ in range(50)];Dh=[demo(r,True) for _ in range(50)]
    if S<=0:
        p=Pol(4,0,True,'l2');x=r.normal(0,1,(4,10));m=MLP(10,16,4,3,0);y=r.normal(0,1,(4,4));o,hs=m.fw(x);u,G=m.grads(hs,2*(o-y));L=lambda:float(((m.fw(x)[0]-y)**2).sum(1).mean())
        for k,q in enumerate(m.P):
            i=np.unravel_index(int(np.argmax(abs(G[k]))),q.shape);e=1e-6;q[i]+=e;a=L();q[i]-=2*e;b=L();q[i]+=e;print(k,G[k][i],(a-b)/2/e)
        print('输入梯度',u[0,0],end=' ');x[0,0]+=1e-6;a=L();x[0,0]-=2e-6;b=L();x[0,0]+=1e-6;print((a-b)/2e-6*4);sys.exit()
    for tag,D,hu in (('script',Ds,False),('human',Dh,True)):
        res={}
        for k in (1,4,16,32,64,128):
            O,A=data(D,k);p=Pol(k,sd,True);p.train(O,A,S,sd+1);a,_=rollout(p,hu,False,900+sd);res[f'k{k}']=a
            if k in(1,16,32):b,_=rollout(p,hu,True,900+sd);res[f'k{k}_te']=b
            print(sd,tag,k,a,res.get(f'k{k}_te'),round(time.time()-t0),flush=True)
            if sd==0 and tag=='human' and k in(1,32):np.savez(f'm_k{k}.npz',**{f'W{i}':w for i,w in enumerate(p.dec.W)},**{f'b{i}':w for i,w in enumerate(p.dec.b)})
        k=32;O,A=data(D,k)
        for nm,c,l in (('nocvae',False,'l1'),('l2',True,'l2'),('l2_nocvae',False,'l2')):
            p=Pol(k,sd,c,l);p.train(O,A,S,sd+1);res[nm]=rollout(p,hu,False,900+sd)[0];print(sd,tag,nm,res[nm],round(time.time()-t0),flush=True)
        # KL 权重敏感性。0.00078 是第一轮的错误取值（L1 求和，KL 固定 .05）在 k=32 时的等效值，保留下来当反面例子
        for bm in (.05/64,.01,.1,1):
            p=Pol(k,sd,True);p.train(O,A,S,sd+1,beta=bm);res[f'beta{bm:g}']=rollout(p,hu,False,900+sd)[0];print(sd,tag,'beta',bm,res[f'beta{bm:g}'],round(time.time()-t0),flush=True)
        out[tag]=res;json.dump(out,open(f'res{sd}.json','w'))
