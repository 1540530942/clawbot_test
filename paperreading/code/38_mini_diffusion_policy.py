#!/usr/bin/env python3
# 第 38 期迷你扩散策略：二维绕障碍，比较直接回归、混合高斯和扩散策略（首轮脚本；诊断与公平对照见证据目录 38b、38c、38d）
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

"""第 38 期玩具实验：二维绕障碍。专家随机从左或右绕过圆形障碍到达目标，所以动作分布有两个峰。
比较：直接回归（MSE）、混合高斯（MDN）、扩散策略（单步、动作序列、不同执行长度、位置或速度控制、延迟）。
观测是最近 2 个位置。只用 NumPy，MLP 代替 U-Net。这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
GOAL=np.array([0.,.8]);OB=np.array([0.,0.]);R=.35;MAXT=70
class MLP:
    def __init__(s,di,H,do,L,seed,last=1e-2):
        r=np.random.default_rng(seed);s.W=[r.normal(0,1/np.sqrt(di),(di,H))]+[r.normal(0,1/np.sqrt(H),(H,H)) for _ in range(L-1)]+[r.normal(0,last,(H,do))];s.b=[np.zeros(H) for _ in range(L)]+[np.zeros(do)];s.P=s.W+s.b;s.M=[np.zeros_like(p) for p in s.P];s.U=[np.zeros_like(p) for p in s.P];s.n=0;s.L=L;s.Em=[p.copy() for p in s.P]
    def fw(s,x):
        hs=[x];h=x
        for i in range(s.L):
            n=np.maximum(h@s.W[i]+s.b[i],0)
            if i>0:n=n+h
            hs.append(n);h=n
        return h@s.W[-1]+s.b[-1],hs
    def step(s,hs,g,lr,ema=.995):
        B=len(g);G=[None]*(2*(s.L+1));G[s.L]=hs[s.L].T@g/B;G[2*s.L+1]=g.mean(0);u=g@s.W[-1].T
        for i in range(s.L-1,-1,-1):
            pre=hs[i];act=(pre@s.W[i]+s.b[i])>0;d=u*act;G[i]=pre.T@d/B;G[s.L+1+i]=d.mean(0);u=d@s.W[i].T+(u if i>0 else 0)
        s.n+=1
        for k,(p,gg) in enumerate(zip(s.P,G)):
            s.M[k]=.9*s.M[k]+.1*gg;s.U[k]=.999*s.U[k]+.001*gg*gg;p-=lr*(s.M[k]/(1-.9**s.n))/(np.sqrt(s.U[k]/(1-.999**s.n))+1e-8);s.Em[k]=ema*s.Em[k]+(1-ema)*p
    def ema(s):
        for p,e in zip(s.P,s.Em):p[...]=e
def expert(r):
    """一条示范：选一边，走平滑的弧"""
    side=1. if r.random()<.5 else -1.;x0=r.uniform(-.25,.25);p=[np.array([x0,-.8])];T=int(r.integers(38,48));wp=[np.array([side*(R+.2+r.uniform(0,.08)),0.]),GOAL];pts=[]
    a,b,c=p[0],wp[0],wp[1]
    for t in range(1,T+1):
        u=t/T;q=(1-u)**2*a+2*u*(1-u)*(2*b-.5*(a+c))+u*u*c;pts.append(q)
    tr=np.array([a]+pts);tr=tr+np.cumsum(r.normal(0,.0015,tr.shape),0);return tr,side
def demos(r,N):
    return [expert(r)[0] for _ in range(N)]
def dataset(D,To,Tp,vel):
    O=[];A=[]
    for tr in D:
        for t in range(len(tr)-1):
            o=np.r_[tr[max(t-To+1,0)],tr[t]];idx=np.minimum(np.arange(t+1,t+1+Tp),len(tr)-1);a=tr[idx]
            if vel:a=a-tr[idx-1]
            O.append(o);A.append(a.reshape(-1))
    return np.array(O),np.array(A)
class Norm:
    def __init__(s,A):
        s.lo,s.hi=A.min(0),A.max(0);s.w=np.maximum(s.hi-s.lo,1e-6)
    def f(s,a):return 2*(a-s.lo)/s.w-1
    def inv(s,a):return (a+1)/2*s.w+s.lo
K=100;f=np.cos((np.arange(K+1)/K+.008)/1.008*np.pi/2)**2;AB=(f/f[0])[1:];
def kemb(k,d=16):
    f_=np.exp(np.linspace(0,np.log(100),d//2));p=(k/K)[:,None]*f_;return np.c_[np.sin(p),np.cos(p)]
class Reg:
    name='reg'
    def __init__(s,O,A,seed):
        s.n=Norm(A);s.Ad=A.shape[1];s.net=MLP(4,128,s.Ad,3,seed)
    def train(s,O,A,steps,seed):
        r=np.random.default_rng(seed);Y=s.n.f(A)
        for st in range(1,steps+1):
            k=r.integers(0,len(O),256);o,hs=s.net.fw(O[k]);s.net.step(hs,2*(o-Y[k]),1e-3*(1 if st<.8*steps else .2))
        s.net.ema()
    def sample(s,O,r):return s.n.inv(s.net.fw(O)[0])
class MDN:
    name='mdn'
    def __init__(s,O,A,seed,Km=5):
        s.n=Norm(A);s.Ad=A.shape[1];s.Km=Km;s.net=MLP(4,128,Km+2*Km*s.Ad,3,seed,.05)
    def parts(s,o):
        pi=o[:,:s.Km];mu=o[:,s.Km:s.Km+s.Km*s.Ad].reshape(-1,s.Km,s.Ad);ls=np.clip(o[:,s.Km+s.Km*s.Ad:].reshape(-1,s.Km,s.Ad),-6,2);return pi,mu,ls
    def train(s,O,A,steps,seed):
        r=np.random.default_rng(seed);Y=s.n.f(A)
        for st in range(1,steps+1):
            k=r.integers(0,len(O),256);o,hs=s.net.fw(O[k]);pi,mu,ls=s.parts(o);y=Y[k];lw=pi-pi.max(1,keepdims=True);lw=lw-np.log(np.exp(lw).sum(1,keepdims=True));e=(y[:,None]-mu)/np.exp(ls);lg=lw-.5*(e*e+2*ls).sum(2);m=lg.max(1,keepdims=True);res=np.exp(lg-m);R_=res/res.sum(1,keepdims=True);G=np.c_[np.exp(lw)-R_,(-R_[:,:,None]*e/np.exp(ls)).reshape(256,-1),(R_[:,:,None]*(1-e*e)).reshape(256,-1)];s.net.step(hs,G,1e-3*(1 if st<.8*steps else .2))
        s.net.ema()
    def sample(s,O,r):
        o=s.net.fw(O)[0];pi,mu,ls=s.parts(o);lw=pi-pi.max(1,keepdims=True);q=np.exp(lw);q/=q.sum(1,keepdims=True);c=(q.cumsum(1)>r.random((len(O),1))).argmax(1);i=np.arange(len(O));return s.n.inv(mu[i,c]+np.exp(ls[i,c])*r.normal(0,1,(len(O),s.Ad)))
class DP:
    name='dp'
    def __init__(s,O,A,seed):
        s.n=Norm(A);s.Ad=A.shape[1];s.net=MLP(4+16+s.Ad,128,s.Ad,3,seed)
    def train(s,O,A,steps,seed):
        r=np.random.default_rng(seed);Y=s.n.f(A)
        for st in range(1,steps+1):
            i=r.integers(0,len(O),256);k=r.integers(0,K,256);e=r.normal(0,1,(256,s.Ad));x=np.sqrt(AB[k])[:,None]*Y[i]+np.sqrt(1-AB[k])[:,None]*e;o,hs=s.net.fw(np.c_[O[i],kemb(k),x]);s.net.step(hs,2*(o-e)/s.Ad*2,1e-3*min(1,st/200)*(1 if st<.8*steps else .2))
        s.net.ema()
    def sample(s,O,r,steps=10):
        """DDIM，10 步"""
        n=len(O);x=r.normal(0,1,(n,s.Ad));ts=np.linspace(K-1,0,steps).astype(int)
        for j,t in enumerate(ts):
            k=np.full(n,t);ep=s.net.fw(np.c_[O,kemb(k),x])[0];x0=np.clip((x-np.sqrt(1-AB[t])*ep)/np.sqrt(AB[t]),-1,1)
            if j<steps-1:
                tp=ts[j+1];x=np.sqrt(AB[tp])*x0+np.sqrt(1-AB[tp])*(x-np.sqrt(AB[t])*x0)/np.sqrt(1-AB[t])
            else:x=x0
        return s.n.inv(x)
def rollout(pol,Tp,Ta,vel,seed,n=100,latency=0,maxt=MAXT):
    """闭环：每 Ta 步重新预测，延迟 latency 步表示观测到动作开始之间的间隔"""
    r=np.random.default_rng(seed);x0=r.uniform(-.25,.25,n);pos=np.c_[x0,np.full(n,-.8)];prev=pos.copy();act=np.zeros(n,bool);done=np.zeros(n,bool);hit=np.zeros(n,bool);tr=[pos.copy()];Q=None
    for t in range(maxt):
        if t%Ta==0:
            O=np.c_[prev,pos];Q=pol.sample(O,r).reshape(n,Tp,2);q0=t
        idx=min(t-q0+latency,Tp-1);a=Q[:,idx];new=pos+a if vel else a;new=np.where((done|hit)[:,None],pos,new);prev=pos;pos=new;tr.append(pos.copy());hit|=np.linalg.norm(pos-OB,axis=1)<R-.03;done|=np.linalg.norm(pos-GOAL,axis=1)<.1
    tr=np.stack(tr,1);succ=done&~hit;xmid=[];flips=0.
    for i in range(n):
        y=tr[i,:,1];j=(np.abs(y)).argmin();xmid.append(tr[i,j,0])
    xm=np.array(xmid);v=np.diff(tr,axis=1);acc=np.diff(v,axis=1);jerk=float(np.sqrt((acc**2).sum(2)).mean());return dict(succ=float(succ.mean()),hit=float(hit.mean()),left=float((xm[succ]<0).mean()) if succ.any() else None,jerk=jerk,len=float(np.sqrt((v**2).sum(2)).sum(1).mean()))
def run(cls,D,Tp,Ta,vel,steps,seed,lat=(0,)):
    O,A=dataset(D,2,Tp,vel);p=cls(O,A,seed);t0=time.time();p.train(O,A,steps,seed+1);res={}
    for l in lat:res[str(l)]=rollout(p,Tp,Ta,vel,900+seed,latency=l)
    return p,res,round(time.time()-t0)
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 5000;out={};t0=time.time();r=np.random.default_rng(sd);D=demos(r,200);ex=[]
    if S<=0:
        n=MLP(6,16,4,3,0);x=r.normal(0,1,(4,6));y=r.normal(0,1,(4,4));o,hs=n.fw(x);g=2*(o-y);G=[None]*8;G[3]=hs[3].T@g/4;G[7]=g.mean(0);u=g@n.W[-1].T
        for i in (2,1,0):
            pre=hs[i];act=(pre@n.W[i]+n.b[i])>0;d=u*act;G[i]=pre.T@d/4;G[4+i]=d.mean(0);u=d@n.W[i].T+(u if i>0 else 0)
        def L():return float(((n.fw(x)[0]-y)**2).sum(1).mean())
        for k,p in enumerate(n.P):
            i=np.unravel_index(int(np.argmax(abs(G[k]))),p.shape);e=1e-6;p[i]+=e;a=L();p[i]-=2*e;b=L();p[i]+=e;print(k,G[k][i],(a-b)/2/e)
        sys.exit()
    # 示范本身的比例
    arms=[('reg1',Reg,1,1,False),('reg16',Reg,16,8,False),('mdn1',MDN,1,1,False),('mdn16',MDN,16,8,False),('dp1',DP,1,1,False),('dp16',DP,16,8,False),('dp16_a1',DP,16,1,False),('dp16_a16',DP,16,16,False),('dp16_v',DP,16,8,True)]
    for k,c,Tp,Ta,vel in arms:
        p,res,sec=run(c,D,Tp,Ta,vel,S,sd,lat=(0,2,4,6) if k in('dp16','dp16_v') else (0,));out[k]=res;print(sd,k,res,sec,round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'))
        if sd==0 and k=='dp16':np.savez('m.npz',**{f'W{i}':w for i,w in enumerate(p.net.W)},**{f'b{i}':w for i,w in enumerate(p.net.b)},lo=p.n.lo,w=p.n.w,AB=AB,demos=np.array([d[:30:2] for d in D[:20]],dtype=object) if False else np.zeros(1))
    for N in (10,30,100):
        p,res,sec=run(DP,D[:N],16,8,False,S,sd);out[f'n{N}']=res;p2,res2,_=run(MDN,D[:N],1,1,False,S,sd);out[f'mdn_n{N}']=res2;print(sd,N,res,res2,round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'))
