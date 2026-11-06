#!/usr/bin/env python3
# 第 31 期 DDPM 加宽版（256 单元，含截断比较）
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

"""第 31 期玩具实验：8×8 手画数字字形上的 DDPM（只用 NumPy，MLP 代替 U-Net，手写反向传播）。
比较：预测 ε、μ、x0；简化目标和变分界权重；步数和噪声表；再量一量比特都花在哪。这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
G=["..XXXX..,.X....X.,X......X,X......X,X......X,X......X,.X....X.,..XXXX..","...XX...,..XXX...,...XX...,...XX...,...XX...,...XX...,...XX...,.XXXXXX.",".XXXXX..,X.....X.,......X.,.....X..,...XX...,..X.....,.X......,XXXXXXXX",".XXXXX..,......X.,......X.,..XXXX..,......X.,......X.,......X.,.XXXXX..",".....X..,....XX..,...X.X..,..X..X..,.X...X..,XXXXXXXX,.....X..,.....X..","XXXXXXX.,X.......,X.......,XXXXXX..,......X.,......X.,X.....X.,.XXXXX..","..XXXX..,.X......,X.......,XXXXXX..,X.....X.,X.....X.,X.....X.,.XXXXX..","XXXXXXXX,......X.,.....X..,....X...,...X....,...X....,..X.....,..X.....",".XXXXX..,X.....X.,X.....X.,.XXXXX..,X.....X.,X.....X.,X.....X.,.XXXXX..",".XXXXX..,X.....X.,X.....X.,.XXXXXX.,......X.,......X.,.....X..,.XXXX..."]
BM=np.array([[[c=='X' for c in row] for row in g.split(',')] for g in G],float);D=64
def data(r,n):
    y=r.integers(0,10,n);x=BM[y].copy()
    for i in range(n):
        if r.random()<.5:x[i]=np.roll(x[i],int(r.integers(-1,2)),1)
    f=r.random(x.shape)<.02;x=np.where(f,1-x,x);return (x*2-1).reshape(n,-1),y
def sched(T,kind='linear'):
    s=1000/T
    if kind=='linear':b=np.linspace(1e-4*s,.02*s,T)
    elif kind=='const':b=np.full(T,.01*s)
    elif kind=='quad':b=np.linspace(np.sqrt(1e-4*s),np.sqrt(.02*s),T)**2
    b=np.minimum(b,.5);a=1-b;ab=np.cumprod(a);return b,a,ab
def temb(t,T,d=32):
    f=np.exp(np.linspace(0,np.log(1000),d//2));p=(t/T)[:,None]*f;return np.concatenate([np.sin(p),np.cos(p)],1)
class Net:
    def __init__(s,H,seed,E=32):
        r=np.random.default_rng(seed);s.H=H;s.W=[r.normal(0,1/np.sqrt(D+E),(D+E,H)),r.normal(0,1/np.sqrt(H),(H,H)),r.normal(0,1/np.sqrt(H),(H,H)),r.normal(0,1e-3,(H,D))];s.b=[np.zeros(H),np.zeros(H),np.zeros(H),np.zeros(D)]
        s.P=s.W+s.b;s.M=[np.zeros_like(p) for p in s.P];s.U=[np.zeros_like(p) for p in s.P];s.E=[p.copy() for p in s.P];s.n=0
    def fw(s,x,e,keep=False):
        z=np.c_[x,e];hs=[z];h=z
        for i in range(3):
            n=np.maximum(h@s.W[i]+s.b[i],0)
            if i>0:n=n+h      # 残差
            hs.append(n);h=n
        o=h@s.W[3]+s.b[3];return (o,hs) if keep else o
    def step(s,hs,dy,lr,ema=.999):
        B=len(dy);G=[None]*8;g=dy;G[3]=hs[3].T@g/B;G[7]=g.mean(0);u=g@s.W[3].T
        for i in (2,1,0):
            pre=hs[i];act=(pre@s.W[i]+s.b[i])>0;d=u*act;G[i]=pre.T@d/B;G[4+i]=d.mean(0);u=d@s.W[i].T+(u if i>0 else 0)
        s.n+=1;G=G;
        for k,(p,gg) in enumerate(zip(s.P,G)):
            s.M[k]=.9*s.M[k]+.1*gg;s.U[k]=.999*s.U[k]+.001*gg*gg;p-=lr*(s.M[k]/(1-.9**s.n))/(np.sqrt(s.U[k]/(1-.999**s.n))+1e-8);s.E[k]=ema*s.E[k]+(1-ema)*p
    def use_ema(s):
        for p,e in zip(s.P,s.E):p[...]=e
def targets(x0,eps,xt,t,b,a,ab,par):
    """把网络输出解释成 ε、μ 或 x0，统一换算回 ε 的预测函数在 to_eps 里"""
    if par=='eps':return eps
    if par=='x0':return x0
    A=ab[t][:,None];at=a[t][:,None];bt=b[t][:,None];Ap=np.r_[1,ab][t][:,None];return (np.sqrt(Ap)*bt/(1-A))*x0+(np.sqrt(at)*(1-Ap)/(1-A))*xt
def to_eps(o,xt,t,b,a,ab,par):
    A=ab[t][:,None]
    if par=='eps':return o
    if par=='x0':return (xt-np.sqrt(A)*o)/np.sqrt(1-A)
    return (xt-np.sqrt(a[t][:,None])*o)*np.sqrt(1-A)/np.maximum(b[t][:,None],1e-9)
def train(net,T,kind,par,weight,steps,seed,lr=1e-3):
    r=np.random.default_rng(seed);b,a,ab=sched(T,kind);Ap=np.r_[1,ab[:-1]];sig2=b.copy();wv=b*b/(2*sig2*a*(1-ab));wv=wv/wv.mean();h=[]
    for st in range(1,steps+1):
        x0,_=data(r,128);t=r.integers(0,T,128);eps=r.normal(0,1,x0.shape);xt=np.sqrt(ab[t])[:,None]*x0+np.sqrt(1-ab[t])[:,None]*eps;tg=targets(x0,eps,xt,t,b,a,ab,par);o,hs=net.fw(xt,temb(t,T),True);w=(wv[t] if weight=='vlb' else np.ones(128))[:,None];e=o-tg;net.step(hs,2*w*e/D*D/ D,lr*(1 if st<.8*steps else .2))
        if st%max(1,steps//10)==0:h.append((st,round(float((e*e).mean()),4)))
    net.use_ema();return h
def sample(net,T,kind,par,n,seed,sigma='beta',trace=None,clip=False):
    r=np.random.default_rng(seed);b,a,ab=sched(T,kind);Ap=np.r_[1,ab[:-1]];bt=(1-Ap)/(1-ab)*b;x=r.normal(0,1,(n,D))
    for t in range(T-1,-1,-1):
        tt=np.full(n,t);ep=to_eps(net.fw(x,temb(tt,T)),x,tt,b,a,ab,par);z=r.normal(0,1,x.shape) if t>0 else 0
        if clip:
            h0=np.clip((x-np.sqrt(1-ab[t])*ep)/np.sqrt(ab[t]),-1,1);ep=(x-np.sqrt(ab[t])*h0)/np.sqrt(1-ab[t])
        if trace is not None and t in trace:trace[t]=((x-np.sqrt(1-ab[t])*ep)/np.sqrt(ab[t])).copy()
        x=(x-b[t]/np.sqrt(1-ab[t])*ep)/np.sqrt(a[t])+(np.sqrt(b[t] if sigma=='beta' else bt[t]))*z
    return x
TPL=(BM*2-1).reshape(10,-1)
def quality(x):
    """最接近模板的比例：像素错 ≤6 个算好样本（允许 1 格平移，模板含三种平移）"""
    A=np.stack([(np.roll(BM,s,2)*2-1).reshape(10,-1) for s in(-1,0,1)]);X=np.sign(x);d=(X[None,None]!=A[:,:,None]).sum(-1);m=d.min(0);lab=m.argmin(0);e=m.min(0);ok=e<=6;cnt=np.bincount(lab[ok],minlength=10);p=cnt/max(1,cnt.sum());H=-(p[p>0]*np.log(p[p>0])).sum();return dict(good=float(ok.mean()),meanerr=float(e.mean()),modes=int((cnt>0).sum()),H=float(H/np.log(10)))
def fid(x,y):
    a,b=x.mean(0),y.mean(0);A=np.cov(x.T)+1e-6*np.eye(D);B=np.cov(y.T)+1e-6*np.eye(D);va,ua=np.linalg.eigh(A);sq=(ua*np.sqrt(np.maximum(va,0)))@ua.T;c=np.linalg.eigvalsh(sq@B@sq);return float(((a-b)**2).sum()+np.trace(A)+np.trace(B)-2*np.sqrt(np.maximum(c,0)).sum())
def rate_dist(net,T,kind,par,seed,n=400):
    """按式 5 逐项算码长（比特/维），再算每步的重建误差（式 15）。对应论文图 5"""
    r=np.random.default_rng(seed);b,a,ab=sched(T,kind);x0,_=data(r,n);L=[];rm=[];Ap=np.r_[1,ab[:-1]]
    for t in range(T):
        tt=np.full(n,t);eps=r.normal(0,1,x0.shape);xt=np.sqrt(ab[t])*x0+np.sqrt(1-ab[t])*eps;ep=to_eps(net.fw(xt,temb(tt,T)),xt,tt,b,a,ab,par);h=(xt-np.sqrt(1-ab[t])*ep)/np.sqrt(ab[t]);rm.append(float(np.sqrt(((h-x0)**2).mean())))
        if t>0:
            mq=np.sqrt(Ap[t])*b[t]/(1-ab[t])*x0+np.sqrt(a[t])*(1-Ap[t])/(1-ab[t])*xt;mp=(xt-b[t]/np.sqrt(1-ab[t])*ep)/np.sqrt(a[t]);vq=(1-Ap[t])/(1-ab[t])*b[t];vp=b[t];kl=.5*(np.log(vp/vq)+vq/vp-1+((mq-mp)**2)/vp);L.append(float(kl.mean()/np.log(2)))
        else:
            mp=(xt-b[0]/np.sqrt(1-ab[0])*ep)/np.sqrt(a[0]);s=np.sqrt(b[0]);L.append(float((.5*((x0-mp)/s)**2+.5*np.log(2*np.pi*b[0])+np.log(1)).mean()/np.log(2)))
    return dict(L=L,rmse=rm,LT=float((.5*(ab[-1]*(x0**2)-(ab[-1])-np.log(1-ab[-1]))).mean()/np.log(2)))
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 12000;H=256;out={};t0=time.time()
    if S<=0:
        net=Net(16,0);r=np.random.default_rng(0);x=r.normal(0,1,(4,D));e=temb(np.array([1,2,3,4]),10);y=r.normal(0,1,(4,D))
        def L():return float(((net.fw(x,e)-y)**2).sum(1).mean())
        o,hs=net.fw(x,e,True);S0=[p.copy() for p in net.P];lr=1e-9;G=[]
        import copy
        B=4;g=2*(o-y);Gs=[None]*8;Gs[3]=hs[3].T@g/B;Gs[7]=g.mean(0);u=g@net.W[3].T
        for i in (2,1,0):
            pre=hs[i];act=(pre@net.W[i]+net.b[i])>0;dd=u*act;Gs[i]=pre.T@dd/B;Gs[4+i]=dd.mean(0);u=dd@net.W[i].T+(u if i>0 else 0)
        for k,p in enumerate(net.P):
            i=np.unravel_index(int(np.argmax(abs(Gs[k]))),p.shape);ee=1e-6;p[i]+=ee;a=L();p[i]-=2*ee;b=L();p[i]+=ee;print(k,Gs[k][i],(a-b)/2/ee)
        sys.exit()
    ref,_=data(np.random.default_rng(99),4000)
    # 诊断：128 单元时 ε 预测合格率只有约 60%，x0 预测 99%。猜测：MLP 没有直连，要把 64 维噪声“抄”到输出，太窄抄不下。这里加宽到 256，并看截断的影响
    arms=[('eps',200,'linear','eps','simple'),('x0',200,'linear','x0','simple'),('eps_vlb',200,'linear','eps','vlb'),('mu',200,'linear','mu','simple')]
    for k,T,kind,par,w in arms:
        net=Net(H,sd);h=train(net,T,kind,par,w,S,sd+1);x=sample(net,T,kind,par,1000,sd+2);q=quality(x);q['fid']=fid(x,ref);res=dict(q=q,hist=h,T=T,sec=round(time.time()-t0));xc=sample(net,T,kind,par,1000,sd+2,clip=True);qc=quality(xc);qc['fid']=fid(xc,ref);res['q_clip']=qc;q['maxabs']=float(np.abs(x).max());q['rmse_mid']=None;print(sd,k,'截断',qc,flush=True);
        if k in('eps','eps_vlb'):res['rd']=rate_dist(net,T,kind,par,sd+3)
        out[k]=res;print(sd,k,q,h[-1],res['sec'],flush=True);json.dump(out,open(f'res{sd}.json','w'))
        if sd==0 and k=='eps':
            np.savez('m.npz',**{f'W{i}':net.W[i] for i in range(4)},**{f'b{i}':net.b[i] for i in range(4)});tr={t:None for t in (199,150,100,75,50,25,10,0)};xs=sample(net,T,kind,par,8,5,trace=tr);np.savez('trace.npz',**{f't{t}':v for t,v in tr.items()},x=xs)
    x=ref[:1000];q=quality(x);q['fid']=fid(x,ref[1000:]);out['real']=q;json.dump(out,open(f'res{sd}.json','w'));print(sd,'真实',q,flush=True)
