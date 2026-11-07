#!/usr/bin/env python3
# 第 23 期 Mamba 选择性复制加长版
# 单文件版：下面依次是极小的自动微分、迷你 Transformer 和实验本身，只依赖 NumPy。
import gc
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
        if s.n%20==0: gc.collect()   # 计算图可能有引用环，定期回收防止内存涨
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

"""第 23 期玩具实验：迷你 Mamba。两个合成任务，比较线性时不变 SSM（S4 风格）、选择性 SSM（S6）和注意力：
选择性复制：长序列里散布几个数据词，其余是噪声，最后按顺序复述（只考查全部词记住）。
归纳头：序列里见过 A B，再看到 A 时预测 B。训练用短序列，再测更长。"""
import sys,json,time,numpy as np
def exp_(t):
    e=np.exp(t.v);r=T(e,(t,));r.bk=lambda g:(g*e,);return r
def softplus(t):
    v=np.logaddexp(0,t.v);s=1/(1+np.exp(-t.v));r=T(v,(t,));r.bk=lambda g:(g*s,);return r
def swish(t):
    s=1/(1+np.exp(-t.v));r=T(t.v*s,(t,));r.bk=lambda g:(g*(s+t.v*s*(1-s)),);return r
def select(t,i):
    """取第 i 个时间步（B,L,...）"""
    r=T(t.v[:,i],(t,))
    def bk(g):
        o=np.zeros_like(t.v);o[:,i]=g;return (o,)
    r.bk=bk;return r
def stack(L):
    r=T(np.stack([x.v for x in L],1),tuple(L));r.bk=lambda g:tuple(g[:,i] for i in range(len(L)));return r
class Block:
    """归一化 → 升维 → 一路过 SSM，另一路做门 → 相乘 → 降维，残差。D→E·D=M 个通道，每通道 N 个状态。"""
    def __init__(s,D,N,sel,r,ps,E=2):
        s.D,s.N,s.M,s.sel=D,N,E*D,sel;M=s.M;lin=lambda a,b:(lambda t:(ps.append(t),t)[1])(T(r.normal(0,1/np.sqrt(a),(a,b))))
        s.U=lin(D,M);s.G=lin(D,M);s.O=lin(M,D)
        s.A=-(np.arange(N)+1.0)   # S4D-Real 初值，这里固定
        s.logd=T(r.uniform(-4,-1,(M,)));ps.append(s.logd)
        if sel:
            s.Bx=lin(M,N);s.Cx=lin(M,N);s.dx=lin(M,1)
        else:
            s.B0=T(r.normal(0,.5,(M,N)));s.C0=T(r.normal(0,.5,(M,N)));ps+= [s.B0,s.C0]
    def forward(s,x,trace=None):
        B,L,_=x.v.shape;g=x.lnorm();u=swish(g@s.U);gate=swish(g@s.G);M,N=s.M,s.N;h=T(np.zeros((B,M,N)));ys=[];A=s.A
        for t in range(L):
            ut=select(u,t)            # (B,M)
            if s.sel:
                d=softplus(s.logd+((ut@s.dx)))        # (B,M)：广播一维
                bt=(ut@s.Bx);ct=(ut@s.Cx)            # (B,N)
                ad=T(d.v[...,None]*A,(d,));ad.bk=(lambda g2,A=A:((g2*A).sum(-1),))
                ab=exp_(ad);bb=ab*(1/A)+(-1/A)         # (A̅−1)/A，乘 B 再乘输入
                inp=T(ut.v[:,:,None]*bt.v[:,None,:],(ut,bt))
                def bk2(g2,bv=bt.v,uv=ut.v):return ((g2*bv[:,None,:]).sum(-1),(g2*uv[:,:,None]).sum(1))
                inp.bk=bk2
                h=ab*h+bb*inp
                y=T((h.v*ct.v[:,None,:]).sum(-1),(h,ct))
                def bk3(g2,hv=h.v,cv=ct.v):return (g2[...,None]*cv[:,None,:],(g2[...,None]*hv).sum(1))
                y.bk=bk3
                if trace is not None:trace.append(d.v[0].mean())
            else:
                d=softplus(s.logd);ad=T(d.v[...,None]*A,(d,));ad.bk=(lambda g2,A=A:((g2*A).sum(-1),));ab=exp_(ad);bb=ab*(1/A)+(-1/A)
                inp=T(ut.v[:,:,None]*s.B0.v[None],(ut,s.B0))
                def bk4(g2,bv=s.B0.v,uv=ut.v):return ((g2*bv[None]).sum(-1),(g2*uv[:,:,None]).sum(0))
                inp.bk=bk4
                h=ab*h+bb*inp
                y=T((h.v*s.C0.v[None]).sum(-1),(h,s.C0))
                def bk5(g2,hv=h.v,cv=s.C0.v):return (g2[...,None]*cv[None],(g2[...,None]*hv).sum(0))
                y.bk=bk5
            ys.append(y)
        Y=stack(ys)
        return x+((Y*gate)@s.O)
class Model:
    def __init__(s,V,sel,seed,D=32,N=4,layers=2):
        r=np.random.default_rng(seed);s.ps=[];s.E=T(r.normal(0,1/np.sqrt(D),(V,D)));s.ps.append(s.E);s.V=V;s.bl=[Block(D,N,sel,r,s.ps) for _ in range(layers)];s.D=D
    def fw(s,x,trace=None):
        h=T(np.eye(s.V)[x])@s.E*np.sqrt(s.D)
        for i,b in enumerate(s.bl):h=b.forward(h,trace if i==0 else None)
        return h.lnorm()@s.E.t(1,0)
NOISE=0;VS=10;KS=4   # 选择性复制
def scopy(r,B,L):
    x=np.zeros((B,L+KS+1),int);y=np.zeros_like(x);w=np.zeros(x.shape,float);GO=VS
    for i in range(B):
        pos=np.sort(r.choice(L,KS,replace=False));d=r.integers(1,VS,KS);x[i,pos]=d;x[i,L]=GO;x[i,L+1:L+1+KS-1]=d[:-1];y[i,L:L+KS]=d;w[i,L:L+KS]=1
    return x[:,:L+KS],y[:,:L+KS],w[:,:L+KS]
VI=12
def induct(r,B,L):
    x=r.integers(1,VI,(B,L));y=np.zeros((B,L),int);w=np.zeros((B,L))
    TRIG=VI
    for i in range(B):
        p=int(r.integers(0,L-3));x[i,p]=TRIG;tar=x[i,p+1];x[i,-1]=TRIG;x[i,x[i]==-1]=1
        # 触发词只出现两次
        rest=np.where(x[i]==TRIG)[0]
        y[i,-1]=tar;w[i,-1]=1
    return x,y,w
def acc(m,task,L,seed,B=32,K=6):
    r=np.random.default_rng(seed);ok=tot=0
    for _ in range(K):
        x,y,w=task(r,B,L);p=(fw_np(m,x) if isinstance(m,Model) else m.fw(x).v).argmax(-1);ok+=((p==y)*w).sum();tot+=w.sum()
    return float(ok/tot)
def _sw(v):
    a=1/(1+np.exp(-v));return v*a
def _ln(v,eps=1e-6):
    d=v-v.mean(-1,keepdims=True);return d/np.sqrt((d*d).mean(-1,keepdims=True)+eps)
def block_np(s,x):
    """和 Block.forward 同样的公式，但不建计算图：长序列评测时省内存（第一轮在 L=1024 内存超限）"""
    B,L,_=x.shape;g=_ln(x);u=_sw(g@s.U.v);gate=_sw(g@s.G.v);h=np.zeros((B,s.M,s.N));A=s.A;Y=np.zeros((B,L,s.M))
    for t in range(L):
        ut=u[:,t]
        if s.sel:
            d=np.logaddexp(0,s.logd.v+ut@s.dx.v);ct=ut@s.Cx.v;ab=np.exp(d[...,None]*A);h=ab*h+(ab/A-1/A)*(ut[:,:,None]*(ut@s.Bx.v)[:,None,:]);Y[:,t]=(h*ct[:,None,:]).sum(-1)
        else:
            d=np.logaddexp(0,s.logd.v);ab=np.exp(d[...,None]*A);h=ab*h+(ab/A-1/A)*(ut[:,:,None]*s.B0.v[None]);Y[:,t]=(h*s.C0.v[None]).sum(-1)
    return x+(Y*gate)@s.O.v
def fw_np(m,x):
    h=np.eye(m.V)[x]@m.E.v*np.sqrt(m.D)
    for b in m.bl:h=block_np(b,h)
    return _ln(h)@m.E.v.T
class Att(Enc):
    def fw(s,x,trace=None):
        return s.lm(x)
def train(m,task,L,steps,r,lr):
    o=Adam(m.ps)
    for st in range(1,steps+1):
        x,y,w=task(r,16,L);v,p,g=ce(m.fw(x),y,w);p.backward(g)
        for q in m.ps:
            if q.g is None:q.g=np.zeros_like(q.v)
        o.step(lr*min(1,st/100)*(.3 if st>.85*steps else 1))
    return m,float(v)
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 1500;t0=time.time();out={}
    if S<=0:
        m=Model(VS+1,True,0,D=8,N=2);r=np.random.default_rng(0);x,y,w=scopy(r,2,12)
        def L(m):return ce(m.fw(x),y,w)[0]
        for name,pp,idx in(('Bx',m.bl[0].Bx,(1,0)),('U',m.bl[0].U,(1,2)),('dx',m.bl[1].dx,(2,0)),('E',m.E,(3,1)),('logd',m.bl[0].logd,(1,)),('Cx',m.bl[0].Cx,(0,1)),('O',m.bl[1].O,(2,3))):
            for p in m.ps:p.g=None
            v,pr,g=ce(m.fw(x),y,w);pr.backward(g);a=pp.g[idx];e=1e-5;pp.v[idx]+=e;c=L(m);pp.v[idx]-=2*e;d=L(m);pp.v[idx]+=e;print(name,a,(c-d)/2/e)
        m2=Model(VS+1,False,0,D=8,N=2)
        for name,pp,idx in(('B0',m2.bl[0].B0,(1,0)),('C0',m2.bl[1].C0,(1,0))):
            for p in m2.ps:p.g=None
            v,pr,g=ce(m2.fw(x),y,w);pr.backward(g);a=pp.g[idx];e=1e-5;pp.v[idx]+=e;c=ce(m2.fw(x),y,w)[0];pp.v[idx]-=2*e;d=ce(m2.fw(x),y,w)[0];pp.v[idx]+=e;print(name,a,(c-d)/2/e)
        t=time.time();train(Model(VS+1,True,0),scopy,32,5,np.random.default_rng(0),1e-3);print('5 steps',time.time()-t);sys.exit()
    Ls=[32,64,128,256,512,1024];Li=[16,32,64,128,256,512]
    # 第一轮 1000 步时选择性复制的损失还很高（S6 为 0.7 到 1.4），很可能没训练够。这里只把复制任务加到更多步
    for task,name,V,trL,tests in((scopy,'copy',VS+1,32,Ls),):
        for kind in('S4','S6','att'):
            r=np.random.default_rng(sd+1)
            if kind=='att':m=Att(V,1100,d=32,h=4,ff=64,N=2,seed=sd,causal=True)
            else:m=Model(V,kind=='S6',sd)
            m,v=train(m,task,trL,S,r,1.5e-3);res=dict(train=v,sec=round(time.time()-t0),acc={L:(acc(m,task,L,sd+70,B=1,K=32) if (kind=='att' and L>=256) else acc(m,task,L,sd+70,B=16 if L>=256 else 32,K=2 if L>=256 else 6)) for L in tests},params=int(sum(p.v.size for p in m.ps)));out[f'{name}/{kind}']=res;print(sd,name,kind,res,flush=True);json.dump(out,open(f'res{sd}.json','w'))
            if sd==0 and kind=='S6':np.savez(f'm_{name}.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)})
