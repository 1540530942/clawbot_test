#!/usr/bin/env python3
# 第 30 期 Switch Transformer 加长版
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

"""第 30 期玩具实验：字符级语言模型（语料是本栏目论文的英文全文）。
把 Transformer 的前馈层换成 N 个专家，每个字符只走一个专家（Switch，k=1），每个字符的计算量不变。
比较：普通 Transformer、Switch 2–32、MoE top-2、不用负载均衡、不同容量系数、路由噪声。这是玩具，不代表论文结果。"""
import gc,sys,re,glob,json,time,numpy as np
CH=" abcdefghijklmnopqrstuvwxyz.,'-0";CI={c:i for i,c in enumerate(CH)};V=len(CH);L=48
def corpus():
    t=' '.join(open(f,encoding='utf-8',errors='ignore').read() for f in sorted(glob.glob('../../papers/*/paper.txt')));t=re.sub(r'[0-9]+','0',t.lower());t=re.sub(r"[^ a-z.,'\-0]+",' ',t);t=re.sub(r' +',' ',t);return np.array([CI[c] for c in t])
def take(x,idx):
    r=T(x.v[idx],(x,))
    def bk(g):
        o=np.zeros_like(x.v);o[idx]=g;return (o,)
    r.bk=bk;return r
def takec(p,idx,i):
    r=T(p.v[idx,i][:,None],(p,))
    def bk(g):
        o=np.zeros_like(p.v);o[idx,i]=g[:,0];return (o,)
    r.bk=bk;return r
def place(y,idx,shape):
    o=np.zeros(shape);o[idx]=y.v;r=T(o,(y,));r.bk=lambda g:(g[idx],);return r
class Sw(Enc):
    def setup(s,E,k=1,cap=None,alpha=.01,noise=0.,seed=0,ff=None):
        r=np.random.default_rng(seed+77);s.Ex,s.k,s.cap,s.alpha,s.noise=E,k,cap,alpha,noise;s.nr=r;s.stats=[];s.aux=[]
        if E>0:
            d=s.d;fw=ff or s.ff;lin=lambda a,b,sc=1.:s.par(r.normal(0,sc/np.sqrt(a),(a,b)))
            for l in s.layers:
                s.ps=[p for p in s.ps if p is not l['a'] and p is not l['ab'] and p is not l['b'] and p is not l['bb']]
                del l['a'],l['ab'],l['b'],l['bb']
                l['r']=lin(d,E,.1);l['ex']=[dict(a=lin(d,fw),ab=s.par(np.zeros(fw)),b=lin(fw,d),bb=s.par(np.zeros(d))) for _ in range(E)]
    def switch(s,l,h,train):
        B,n,d=h.v.shape;x=h.rs(B*n,d);p=(x@l['r']).softmax();E=s.Ex;Tn=B*n;Z=p.v
        if train and s.noise>0:
            xn=x.v*s.nr.uniform(1-s.noise,1+s.noise,x.v.shape);Z=(xn@l['r'].v);Z=Z-Z.max(1,keepdims=True)
        top=np.argsort(-Z,1)[:,:s.k];y=None;f=np.zeros(E);drop=0
        for i in range(E):
            idx=np.where((top==i).any(1))[0];f[i]=(top[:,0]==i).mean()
            cap=int(s.cap*Tn/E) if s.cap else Tn
            if len(idx)>cap:drop+=len(idx)-cap;idx=idx[:cap]
            if len(idx)==0:continue
            e=l['ex'][i];o=(((take(x,idx)@e['a'])+e['ab']).relu()@e['b'])+e['bb'];o=o*takec(p,idx,i);o=place(o,idx,(Tn,d));y=o if y is None else y+o
        if train:s.aux.append((p,s.alpha*E*f[None,:]/Tn))
        s.stats.append((f,drop/Tn/s.k,top[:,0].reshape(B,n)));return y.rs(B,n,d)
    def encode2(s,xi,train=True):
        n=xi.shape[1];h=T(np.eye(s.V)[xi])@s.E*np.sqrt(s.d)+T(np.eye(s.L)[:n])@s.P;s.stats=[];s.aux=[]
        for l in s.layers:
            h=(h+s.attn(l,h)).lnorm()
            if s.Ex>0:h=(h+s.switch(l,h,train)).lnorm()
            else:h=(h+(((h@l['a'])+l['ab']).relu()@l['b'])+l['bb']).lnorm()
        return h
    def lm2(s,x,train=True):return s.encode2(x,train)@s.E.t(1,0)
def batch(D,r,B):
    i=r.integers(0,len(D)-L-1,B);return np.stack([D[k:k+L+1] for k in i])
def make(E,seed,**kw):
    m=Sw(V,L,d=48,h=4,ff=64,N=2,seed=seed,causal=True);m.setup(E,seed=seed,**kw);return m
def train(m,Tr,steps,seed):
    r=np.random.default_rng(seed+1);o=Adam(m.ps);t0=time.time();h=[]
    for st in range(1,steps+1):
        x=batch(Tr,r,24);lg=m.lm2(x[:,:-1]);v,p,g=ce(lg,x[:,1:])
        nodes=[(p,g)]+m.aux;root=T(0.,tuple(a for a,_ in nodes));seeds=[b for _,b in nodes];root.bk=lambda gg:tuple(seeds);root.backward()
        o.step(2e-3*min(1,st/100)*(.3 if st>.8*steps else 1))
        if st%10==0:gc.collect()   # 第一轮内存超限：计算图有引用环，靠自动回收来不及
        if st%max(1,steps//8)==0:h.append((st,round(float(v),3)))
    return h,(time.time()-t0)/steps
def evaluate(m,Te,seed=5,n=20):
    r=np.random.default_rng(seed);ls=[];fs=[];dr=[]
    for _ in range(n):
        x=batch(Te,r,24);v,_,_=ce(m.lm2(x[:,:-1],False),x[:,1:]);ls.append(v)
        if m.Ex:fs.append(np.mean([q[0] for q in m.stats],0));dr.append(np.mean([q[1] for q in m.stats]))
    res=dict(nll=float(np.mean(ls)))
    if m.Ex:f=np.mean(fs,0);res.update(maxf=float(f.max()*m.Ex),dead=int((f<1e-3).sum()),drop=float(np.mean(dr)),H=float(-(f[f>0]*np.log(f[f>0])).sum()))
    return res
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 1500
    if S<=0:
        D=corpus();m=make(3,0,cap=2.);r=np.random.default_rng(0);x=batch(D,r,3)[:,:12]
        def L_():
            lg=m.lm2(x[:,:-1]);v,p,g=ce(lg,x[:,1:]);a=0.
            for pp,sdv in m.aux:a=a+float((pp.v*sdv).sum())
            return v+a,p,g
        v,p,g=L_();nodes=[(p,g)]+m.aux;root=T(0.,tuple(a for a,_ in nodes));seeds=[b for _,b in nodes];root.bk=lambda gg:tuple(seeds);root.backward();l=m.layers[0];bad=0
        for nm,par in (('r',l['r']),('ea',l['ex'][1]['a']),('eb',l['ex'][1]['b']),('q',l['q']),('E',m.E)):
            G=par.g.copy();i=np.unravel_index(int(np.argmax(abs(G))),G.shape);e=1e-6;par.v[i]+=e;a=L_()[0];par.v[i]-=2*e;b=L_()[0];par.v[i]+=e;print(nm,G[i],(a-b)/2/e)
        sys.exit()
    D=corpus();cut=int(len(D)*.9);Tr,Te=D[:cut],D[cut:];print('语料',len(D),flush=True);out={};t0=time.time()
    arms=[('dense',dict(E=0)),('sw2',dict(E=2,cap=2.)),('sw4',dict(E=4,cap=2.)),('sw8',dict(E=8,cap=2.)),('sw16',dict(E=16,cap=2.)),('sw32',dict(E=32,cap=2.)),('moe8',dict(E=8,k=2,cap=2.)),('sw8_a0',dict(E=8,cap=2.,alpha=0.)),('sw8_c1',dict(E=8,cap=1.)),('sw8_c125',dict(E=8,cap=1.25)),('sw8_a0c1',dict(E=8,cap=1.,alpha=0.)),('sw8_j',dict(E=8,cap=2.,noise=.01)),('big',dict(E=0,big=1))]
    arms=[a for a in arms if a[0] in('dense','big','sw8','sw32','sw8_a0')]   # 第一轮 1200 步时，所有配置的损失都还在 2.2 附近，没有收敛；这里只拿 5 组跑更久
    for k,kw in arms:
        kw=dict(kw);big=kw.pop('big',0);E=kw.pop('E')
        if big:
            m=Sw(V,L,d=48,h=4,ff=128,N=2,seed=sd,causal=True);m.setup(0,seed=sd)
        else:m=make(E,sd,**kw)
        h,sp=train(m,Tr,S,sd);r=evaluate(m,Te);r.update(hist=h,sec=sp,np=int(sum(p.v.size for p in m.ps)));out[k]=r;print(sd,k,{a:(round(b,3) if isinstance(b,float) else b) for a,b in r.items() if a!='hist'},round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
        if sd==0 and k=='sw8':
            A={}
            for n,p in (('E',m.E),('P',m.P)):A[n]=p.v
            for i,l in enumerate(m.layers):
                for q in 'qkvo':A[f'l{i}_{q}']=l[q].v
                A[f'l{i}_r']=l['r'].v
                for j,e in enumerate(l['ex']):
                    for q in ('a','ab','b','bb'):A[f'l{i}_e{j}_{q}']=e[q].v
            np.savez('m.npz',**A)
