#!/usr/bin/env python3
# 第 29 期迷你实验：手画字形上的知识蒸馏（温度、没见过的数字、少数据）。
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

"""第 29 期玩具实验：用手画数字字形（16×16，随机平移和噪声）代替 MNIST，从头写 MLP 和蒸馏。
大模型用抖动增强和 dropout；小模型比较硬标签和不同温度的蒸馏；再试“迁移集里没有 3”。这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
G=["..XXXX..,.X....X.,X......X,X......X,X......X,X......X,.X....X.,..XXXX..","...XX...,..XXX...,...XX...,...XX...,...XX...,...XX...,...XX...,.XXXXXX.",".XXXXX..,X.....X.,......X.,.....X..,...XX...,..X.....,.X......,XXXXXXXX",".XXXXX..,......X.,......X.,..XXXX..,......X.,......X.,......X.,.XXXXX..",".....X..,....XX..,...X.X..,..X..X..,.X...X..,XXXXXXXX,.....X..,.....X..","XXXXXXX.,X.......,X.......,XXXXXX..,......X.,......X.,X.....X.,.XXXXX..","..XXXX..,.X......,X.......,XXXXXX..,X.....X.,X.....X.,X.....X.,.XXXXX..","XXXXXXXX,......X.,.....X..,....X...,...X....,...X....,..X.....,..X.....",".XXXXX..,X.....X.,X.....X.,.XXXXX..,X.....X.,X.....X.,X.....X.,.XXXXX..",".XXXXX..,X.....X.,X.....X.,.XXXXXX.,......X.,......X.,.....X..,.XXXX..."]
BM=np.array([[[c=='X' for c in row] for row in g.split(',')] for g in G],float)
S=16
def draw(r,y,jit=2):
    """字形放大到不同比例、平移、断笔、加噪声"""
    B=len(y);o=np.zeros((B,S,S))
    for i,t in enumerate(y):
        g=BM[t];k=r.integers(10,13);ix=(np.arange(k)*8//k);im=g[ix][:,ix];dx,dy=r.integers(0,S-k+1,2)
        o[i,dy:dy+k,dx:dx+k]=im*r.uniform(.6,1)
        sh=r.integers(-1,2,S);
        for row in range(S):
            if r.random()<.12:o[i,row]=np.roll(o[i,row],sh[row])
    o=o*(r.random(o.shape)>.1)+r.normal(0,.28,o.shape);return o.reshape(B,-1)
def jitter(X,r,J=2):
    A=X.reshape(-1,S,S).copy()
    for i in range(len(A)):
        dx,dy=r.integers(-J,J+1,2);A[i]=np.roll(np.roll(A[i],dx,1),dy,0)
    return A.reshape(len(X),-1)
class MLP:
    def __init__(s,sz,seed):
        r=np.random.default_rng(seed);s.W=[r.normal(0,1/np.sqrt(a),(a,b)) for a,b in zip(sz,sz[1:])];s.b=[np.zeros(b) for b in sz[1:]];s.r=r
    def fw(s,X,drop=0.):
        H=[X];A=[];h=X
        for i,(W,b) in enumerate(zip(s.W,s.b)):
            z=h@W+b
            if i<len(s.W)-1:
                h=np.maximum(z,0)
                if drop>0:
                    m=(s.r.random(h.shape)>drop)/(1-drop);h=h*m;A.append(m)
                else:A.append(None)
                H.append(h)
            else:h=z
        return h,H,A
    def bk(s,H,A,dz):
        G=[]; g=dz
        for i in range(len(s.W)-1,-1,-1):
            G.append((H[i].T@g/len(g),g.mean(0)))
            if i>0:
                u=g@s.W[i].T
                if A[i-1] is not None:u=u*A[i-1]
                g=u*(H[i]>0)
        return G[::-1]
def softmax(z,T=1.):
    z=z/T;z=z-z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
def train(m,X,y,steps,lr=2e-3,B=128,drop=0.,jit=False,soft=None,T=1.,alpha=.1,seed=0):
    """soft: 软目标 logit（老师）。alpha：硬标签权重。软损失乘 T²，和论文一样"""
    r=np.random.default_rng(seed+5);n=len(X);P=m.W+m.b;Mo=[np.zeros_like(p) for p in P];U=[np.zeros_like(p) for p in P]
    for st in range(1,steps+1):
        k=r.integers(0,n,B);x=X[k]
        if jit:x=jitter(x,r)
        z,H,A=m.fw(x,drop);g=np.zeros_like(z)
        if y is not None and alpha>0:
            q=softmax(z);e=q.copy();e[np.arange(B),y[k]]-=1;g=g+alpha*e
        if soft is not None:
            g=g+(softmax(z,T)-softmax(soft[k],T))*T   # dC/dz=(q-p)/T，再乘 T²
        G=m.bk(H,A,g);L=[a for a,_ in G]+[b for _,b in G]
        for i,(p,d) in enumerate(zip(P,L)):
            Mo[i]=.9*Mo[i]+.1*d;U[i]=.999*U[i]+.001*d*d;p-=lr*(1 if st<.7*steps else .2)*(Mo[i]/(1-.9**st))/(np.sqrt(U[i]/(1-.999**st))+1e-8)
    return m
def errs(m,X,y,bias=None):
    z=m.fw(X)[0]
    if bias is not None:z=z+bias
    return int((z.argmax(1)!=y).sum())
if __name__=='__main__':
    sd=int(sys.argv[1]);out={};t0=time.time();r=np.random.default_rng(sd)
    N=int(sys.argv[2]) if len(sys.argv)>2 else 1500;St=int(sys.argv[3]) if len(sys.argv)>3 else 3000
    yt=r.integers(0,10,N);Xt=draw(r,yt);ye=r.integers(0,10,4000);Xe=draw(r,ye)
    Big=MLP([256,128,128,10],sd);train(Big,Xt,yt,St,drop=.3,jit=True,seed=sd);out['teacher']=errs(Big,Xe,ye);print(sd,'老师错',out['teacher'],'/4000',round(time.time()-t0),flush=True)
    soft=Big.fw(Xt)[0]
    res={}
    for H in (8,16,32,64):
        base=MLP([256,H,H,10],sd+1);train(base,Xt,yt,St,seed=sd);res[f'{H}/hard']=errs(base,Xe,ye)
        for T in (1,2,4,8,20):
            s=MLP([256,H,H,10],sd+1);train(s,Xt,yt,St,soft=soft,T=T,seed=sd);res[f'{H}/T{T}']=errs(s,Xe,ye)
        print(sd,H,{k.split('/')[1]:v for k,v in res.items() if k.startswith(f'{H}/')},round(time.time()-t0),flush=True)
    out['res']=res;H=64
    # 没有 3 的迁移集（论文 §3）
    k=yt!=3;s=MLP([256,H,H,10],sd+1);train(s,Xt[k],None,St,soft=soft[k],T=8,alpha=0,seed=sd);e0=errs(s,Xe,ye);t3=ye==3;e3=errs(s,Xe[t3],ye[t3]);best=(e0,0.)
    for b in np.arange(0,12.1,.5):
        bi=np.zeros(10);bi[3]=b;e=errs(s,Xe,ye,bi)
        if e<best[0]:best=(e,float(b))
    bi=np.zeros(10);bi[3]=best[1];out['no3']=dict(total=e0,on3=e3,n3=int(t3.sum()),bias=best[1],total_b=best[0],on3_b=errs(s,Xe[t3],ye[t3],bi))
    hd=MLP([256,H,H,10],sd+1);train(hd,Xt[k],yt[k],St,seed=sd);out['no3']['hard_on3']=errs(hd,Xe[t3],ye[t3]);out['no3']['hard_total']=errs(hd,Xe,ye);print(sd,out['no3'],flush=True)
    # 少量数据（论文表 5）：3% 的训练数据
    frac={}
    for f in (.03,.1,.3,1.):
        n=int(N*f);a=MLP([256,H,H,10],sd+2);train(a,Xt[:n],yt[:n],St,seed=sd);b=MLP([256,H,H,10],sd+2);train(b,Xt[:n],yt[:n],St,soft=soft[:n],T=8,seed=sd);frac[f]=dict(hard=errs(a,Xe,ye),soft=errs(b,Xe,ye))
    out['frac']=frac;print(sd,frac,round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
    if sd==0:np.savez('m.npz',W0=Big.W[0],W1=Big.W[1],W2=Big.W[2],b0=Big.b[0],b1=Big.b[1],b2=Big.b[2]);np.savez('ex.npz',X=Xe[:40],y=ye[:40])
