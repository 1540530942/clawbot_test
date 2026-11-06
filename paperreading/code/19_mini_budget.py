#!/usr/bin/env python3
# 第 19 期 LLaMA 玩具实验：模型大小和读多少的账，SwiGLU 和 ReLU
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

"""第 19 期玩具实验。语言同第 15 期（二阶马尔可夫，数据不重复）。
1 算力预算相同，不同大小的模型谁好；继续喂更多数据，小模型能否追上大模型的成绩（论文的“推理预算”想法）。
2 同样参数量，ReLU 前馈和 SwiGLU 前馈比较。"""
import sys,json,time,numpy as np
V=16;L=24;B=32
RNG=np.random.default_rng(12345);TAB=RNG.integers(0,V,(V,V,3));W3=RNG.dirichlet([1,1,1],(V,V));ENT=float(np.mean(-(W3*np.log(W3)).sum(-1)))
def gen(r,n,l=L+1):
    x=np.zeros((n,l),int);x[:,:2]=r.integers(0,V,(n,2))
    for t in range(2,l):
        a,b=x[:,t-2],x[:,t-1];c=(W3[a,b].cumsum(1)>r.random((n,1))).argmax(1);x[:,t]=TAB[a,b,c]
    return x
def swish(t):
    s=1/(1+np.exp(-t.v));r=T(t.v*s,(t,));r.bk=lambda g:(g*(s+t.v*s*(1-s)),);return r
class G(Enc):
    """SwiGLU：swish(W x)·(V x) 再线性投影；隐藏维 ≈ 2/3·4d"""
    def __init__(s,d,seed,glu):
        ff=2*d if not glu else max(2,int(4*d/3));super().__init__(V,L,d=d,h=max(2,d//8),ff=ff,N=2,seed=seed,causal=True);s.glu=glu
        if glu:
            r=np.random.default_rng(seed+7)
            for l in s.layers:l['g']=T(r.normal(0,1/np.sqrt(d),(d,ff)));s.ps.append(l['g'])
    def encode(s,x):
        if not s.glu:return super().encode(x)
        n=x.shape[1];h=T(np.eye(s.V)[x])@s.E*np.sqrt(s.d)+T(np.eye(s.L)[:n])@s.P
        for l in s.layers:
            h=(h+s.attn(l,h)).lnorm();a=swish(h@l['a']);b=h@l['g'];h=(h+((a*b)@l['b'])).lnorm()
        return h
def test(m):
    r=np.random.default_rng(777);o=[]
    for _ in range(6):
        x=gen(r,100);o.append(ce(m.lm(x[:,:-1]),x[:,1:])[0])
    return float(np.mean(o))
def nparam(m):return int(sum(p.v.size for p in m.ps))
def run(d,steps,seed,glu=False,every=None):
    m=G(d,seed,glu);o=Adam(m.ps);r=np.random.default_rng(seed+100);h=[];t=time.time();every=every or steps//6
    for st in range(1,steps+1):
        x=gen(r,B);v,p,g=ce(m.lm(x[:,:-1]),x[:,1:]);p.backward(g)
        for q in m.ps:
            if q.g is None:q.g=np.zeros_like(q.v)
        o.step(2e-3*min(1,st/100)*(.1+.9*.5*(1+np.cos(np.pi*st/steps))))
        if st%every==0:h.append((st*B*L,round(test(m),4)))
    return dict(d=d,glu=glu,N=nparam(m),steps=steps,hist=h,loss=h[-1][1],sec=round(time.time()-t))
if __name__=='__main__':
    sd=int(sys.argv[1]);f=float(sys.argv[2]) if len(sys.argv)>2 else 1.
    if f<=0:print(ENT,run(16,20,0,True,10));sys.exit()
    out=dict(ent=ENT,size=[],act=[])
    for d,S in((12,8000),(16,6000),(24,3500),(32,2200),(48,1100)):
        res=run(d,int(S*f),sd);out['size'].append(res);print(sd,d,res['N'],res['loss'],res['sec'],flush=True);json.dump(out,open(f'res{sd}.json','w'))
    for d,glu in((24,False),(24,True),(32,False),(32,True)):
        res=run(d,int(2500*f),sd+11,glu);out['act'].append(res);print(sd,'act',d,glu,res['N'],res['loss'],res['sec'],flush=True);json.dump(out,open(f'res{sd}.json','w'))
