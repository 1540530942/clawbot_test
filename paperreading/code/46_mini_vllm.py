#!/usr/bin/env python3
# 第 46 期玩具实验：服务模拟器
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

"""第 46 期玩具实验：离散事件模拟大模型服务。没有神经网络，只模拟显存和调度。
长度分布、每步耗时都是我们假设的（按论文给的比例 8.4 倍、5.8 倍设定两种负载），KV 槽位按论文 OPT-13B 的 15.7K。
内存：Orca 风格用连续空间并按 2 的幂取整（外部碎片不计，对它有利）；vLLM 风格按块分配，内存不足时抢占最晚到的请求。这是玩具。"""
import sys,json,time,math,numpy as np
SLOTS=15700
def pow2(n):return 1<<math.ceil(math.log2(max(n,1)))
def lens(r,N,kind):
    """对数正态，均值按论文比例。截断到 2048 总长"""
    a,b=(160.,340.) if kind=='share' else (160/8.4,340/5.8)
    p=np.exp(r.normal(np.log(a)-.5,1.,N));o=np.exp(r.normal(np.log(b)-.5,1.,N));p=np.clip(p,4,1024).astype(int);o=np.clip(o,1,2048-p).astype(int);return p,o
def iter_time(b,prompt_tokens):
    """一步的耗时（毫秒）：常数 + 正比批大小 + 提示词阶段的并行计算。假设"""
    return 25+.6*b+.02*prompt_tokens
def simulate(rate,kind,sysm,seed,N=1500,B=16,slots=SLOTS,recover='recompute',samples=1):
    r=np.random.default_rng(seed);P,O=lens(r,N,kind);arr=np.cumsum(r.exponential(1000./rate,N));t=0.;i=0;run=[];lat=np.zeros(N);peak=0;batched=[];used_stat=[];waiting=[];preempt=0
    def alloc(k):
        n,m=P[k],O[k]
        if sysm=='max':return pow2(2048)
        if sysm=='pow2':return pow2(n+pow2(m))
        if sysm=='oracle':return pow2(n+m)
    Q=[];res=0;done=0;blocks=lambda ln:math.ceil(ln/B)*B
    while done<N:
        while i<N and arr[i]<=t:Q.append(i);i+=1
        if not run and not Q:
            t=arr[i];continue
        ptok=0
        if sysm in('max','pow2','oracle'):
            while Q and res+alloc(Q[0])<=slots:
                k=Q.pop(0);res+=alloc(k);run.append([k,P[k],0]);ptok+=P[k]
        elif sysm=='static':
            if not run:
                bs=max(1,slots//2048)
                for _ in range(min(bs,len(Q))):
                    k=Q.pop(0);res+=2048;run.append([k,P[k],0]);ptok+=P[k]
        else:
            use=sum(blocks((e[1]+e[2])) for e in run)
            while Q and use+blocks(P[Q[0]])*samples<=slots*.97:
                k=Q.pop(0);run.append([k,P[k],0]);use+=blocks(P[k])*samples;ptok+=P[k]
        b=len(run);tk=iter_time(b*samples,ptok);t+=tk;batched.append(b);dn=[]
        for e in run:
            e[2]+=1
            if e[2]>=O[e[0]]:dn.append(e)
        for e in dn:
            lat[e[0]]=t-arr[e[0]];done+=1;run.remove(e)
            if sysm in('max','pow2','oracle'):res-=alloc(e[0])
            elif sysm=='static':res-=2048
        if sysm=='static' and run:
            pass
        if sysm in('page',):
            # 空间不够时，抢占最晚到的
            use=sum(blocks(e[1]+e[2]) for e in run)
            while use>slots and len(run)>1:
                e=run.pop();preempt+=1;use-=blocks(e[1]+e[2]);Q.insert(0,e[0]);P[e[0]]+=e[2];O[e[0]]-=e[2]
                if recover=='swap':t+=.002*blocks(e[1]+e[2])
            used_stat.append((sum(e[1]+e[2] for e in run),use))
        else:
            used_stat.append((sum(e[1]+e[2] for e in run),res))
    v=np.array(used_stat[len(used_stat)//4:],float);return dict(nlat=float((lat/np.maximum(O0(seed,N,kind),1)).mean()),batch=float(np.mean(batched)),effective=float((v[:,0]/np.maximum(v[:,1],1)).mean()),preempt=preempt)
def O0(seed,N,kind):
    r=np.random.default_rng(seed);_,o=lens(r,N,kind);return o
def capacity(res,rates,lim=100.):
    """归一化延迟低于 lim 毫秒/词元的最大请求率（线性插值）"""
    c=0
    for a,b in zip(rates,res):
        if b<lim:c=a
    return c
def sharing(seed,k,T,n,B,mode):
    """按块统计节省比例。mode='parallel'：只共享提示词；mode='beam'：候选随机合并"""
    r=np.random.default_rng(seed)
    if mode=='parallel':
        total=k*math.ceil((n+T)/B);shr=math.ceil(n/B)-1 if n>B else 0;return shr*(k-1)/total if total else 0
    seqs=[[-1]*math.ceil(n/B) for _ in range(k)];nid=0;u=0;v=0;ln=n
    for t in range(T):
        keep=r.dirichlet(np.ones(k)*.5);par=r.choice(k,k,p=keep);seqs=[list(seqs[p]) for p in par]
        ln+=1
        if ln%B==1:
            for s in seqs:
                nid+=1;s.append(nid)
        pos=set()
        for s in seqs:
            for j,b in enumerate(s):pos.add((j,b))
        u+=sum(len(s) for s in seqs);v+=len(pos)
    return 1-v/u
if __name__=='__main__':
    sd=int(sys.argv[1]);N=int(sys.argv[2]) if len(sys.argv)>2 else 1500;t0=time.time();res={}
    RT={'share':(.25,.5,1,1.5,2,3,4,6),'alp':(2,4,8,12,16,24,32,48,64)}
    for kind,rates in RT.items():
        rows={}
        for s in ('static','max','pow2','oracle','page'):
            rows[s]=[simulate(q,kind,s,sd*100+7,N) for q in rates];print(kind,s,[round(x['nlat']) for x in rows[s]],round(time.time()-t0),flush=True)
        res[kind]=dict(rates=rates,rows=rows,cap={s:capacity([x['nlat'] for x in rows[s]],rates) for s in rows});json.dump(res,open(f'res{sd}.json','w'))
    res['block']={str(B):simulate(1.,'share','page',sd*100+7,N,B=B) for B in (1,4,16,64,256)};res['block_alp']={str(B):simulate(12.,'alp','page',sd*100+7,N,B=B) for B in (1,4,16,64,256)};print(res['block'],flush=True)
    res['rec']={k:simulate(2.,'share','page',sd*100+7,N,recover=k) for k in ('recompute','swap')}
    res['mem']={str(sl):{s:simulate(1.5,'share',s,sd*100+7,N,slots=sl)['nlat'] for s in ('oracle','page')} for sl in (6000,15700,30000,60000)};print(res['mem'],flush=True)
    res['par']={str(k):float(np.mean([sharing(sd+i,k,int(O0(sd,1,'alp')[0])+60,int(P0),16,'parallel') for i,P0 in enumerate(lens(np.random.default_rng(sd),40,'share')[0])])) for k in (2,4,6)}
    res['beam']={str(k):float(np.mean([sharing(sd+i,k,120,40,16,'beam') for i in range(30)])) for k in (2,4,6)}
    res['sec']=time.time()-t0;json.dump(res,open(f'res{sd}.json','w'));print('done',res['par'],res['beam'],round(time.time()-t0),flush=True)
