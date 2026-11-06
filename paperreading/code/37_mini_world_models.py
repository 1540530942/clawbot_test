#!/usr/bin/env python3
# 迷你世界模型：躲火球游戏，VAE、LSTM+混合密度网络、线性控制器，梦里训练再回真实环境
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

"""第 37 期玩具实验：迷你世界模型。环境是自己写的“躲火球”（像素观测）。
V：MLP-VAE；M：LSTM + 混合密度网络（手写反向传播）；C：线性控制器，用对角 CEM 代替 CMA-ES。
在梦里（M 生成的潜空间）训练，再到真环境测试，比较不同温度 τ。这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
H,Wd=8,9;DIM=H*Wd;ZD=12;KM=3;RN=64;MAXT=300;P_FIRE=.3
class Env:
    """批量环境。网格 H×W，agent 在底行，火球每步下落一格"""
    def __init__(s,n,r):
        s.n,s.r=n,r;s.x=r.integers(0,Wd,n);s.f=np.zeros((n,H,Wd),bool);s.alive=np.ones(n,bool);s.t=0
    def obs(s):
        # 以自己为中心的视角，世界左右相接（像 CarRacing 跟着车走的摄像头）：人永远在底行正中，不用画
        idx=(s.x[:,None]+np.arange(Wd)-Wd//2)%Wd;return s.f[np.arange(s.n)[:,None],:,idx].transpose(0,2,1).astype(float).reshape(s.n,DIM)
    def step(s,a):
        s.x=(s.x+a-1)%Wd;s.f[:,1:]=s.f[:,:-1];s.f[:,0]=False;sp=s.r.random(s.n)<P_FIRE;s.f[np.arange(s.n)[sp],0,s.r.integers(0,Wd,sp.sum())]=True;hit=s.f[np.arange(s.n),H-1,s.x];s.alive&=~hit;s.t+=1;return hit
class MLP:
    def __init__(s,sz,seed,last=None):
        r=np.random.default_rng(seed);s.W=[r.normal(0,1/np.sqrt(a),(a,b)) for a,b in zip(sz,sz[1:])];s.b=[np.zeros(b) for b in sz[1:]];s.P=s.W+s.b;s.M=[np.zeros_like(p) for p in s.P];s.U=[np.zeros_like(p) for p in s.P];s.n=0
    def fw(s,x):
        hs=[x];h=x
        for i in range(len(s.W)-1):
            h=np.maximum(h@s.W[i]+s.b[i],0);hs.append(h)
        return h@s.W[-1]+s.b[-1],hs
    def grads(s,hs,g):
        L=len(s.W);B=len(g);G=[None]*(2*L);G[L-1]=hs[L-1].T@g/B;G[2*L-1]=g.mean(0);u=g@s.W[L-1].T
        for i in range(L-2,-1,-1):
            act=(hs[i]@s.W[i]+s.b[i])>0;d=u*act;G[i]=hs[i].T@d/B;G[L+i]=d.mean(0);u=d@s.W[i].T
        return u,G
    def upd(s,G,lr):
        s.n+=1
        for k,(p,g) in enumerate(zip(s.P,G)):
            s.M[k]=.9*s.M[k]+.1*g;s.U[k]=.999*s.U[k]+.001*g*g;p-=lr*(s.M[k]/(1-.9**s.n))/(np.sqrt(s.U[k]/(1-.999**s.n))+1e-8)
sig=lambda z:1/(1+np.exp(-np.clip(z,-30,30)))
class VAE:
    def __init__(s,seed):
        s.E=MLP([DIM,128,2*ZD],seed);s.D=MLP([ZD,128,DIM],seed+1)
    def enc(s,x):
        o=s.E.fw(x)[0];return o[:,:ZD],np.clip(o[:,ZD:],-6,2)
    def dec(s,z):return sig(s.D.fw(z)[0])
    def train(s,X,r,steps,beta=.05):
        for st in range(1,steps+1):
            x=X[r.integers(0,len(X),128)];o,he=s.E.fw(x);mu,lv=o[:,:ZD],np.clip(o[:,ZD:],-6,2);e=r.normal(0,1,mu.shape);z=mu+np.exp(.5*lv)*e;lg,hd=s.D.fw(z);p=sig(lg);dg=p-x;uz,Gd=s.D.grads(hd,dg);go=np.c_[uz+beta*mu,uz*np.exp(.5*lv)*e*.5+beta*.5*(np.exp(lv)-1)];_,Ge=s.E.grads(he,go);lr=2e-3*(1 if st<.8*steps else .2);s.D.upd(Gd,lr);s.E.upd(Ge,lr)
        x=X[:2000];mu,_=s.enc(x);return float(((s.dec(mu)>.5)!=(x>.5)).mean())
class M:
    """LSTM + MDN，手写时间反向传播。输入 z,a；输出下一步 z 的混合高斯和 done"""
    def __init__(s,seed):
        r=np.random.default_rng(seed);di=ZD+3;s.W=r.uniform(-.08,.08,(di+RN,4*RN));s.b=r.uniform(-.08,.08,4*RN);s.b[:RN]=1.;s.O=r.normal(0,.05,(RN,KM+2*KM*ZD+1));s.ob=np.zeros(KM+2*KM*ZD+1);s.P=[s.W,s.b,s.O,s.ob];s.Mo=[np.zeros_like(p) for p in s.P];s.U=[np.zeros_like(p) for p in s.P];s.n=0
    def step(s,z,a,h,c):
        x=np.c_[z,np.eye(3)[a],h];o=x@s.W+s.b;n=RN;f,i,oo,g=sig(o[:,:n]),sig(o[:,n:2*n]),sig(o[:,2*n:3*n]),np.tanh(o[:,3*n:]);C=f*c+i*g;t=np.tanh(C);Hn=oo*t;return Hn,C,(x,f,i,oo,g,C,t,c)
    def split(s,out):
        return out[:,:KM],out[:,KM:KM+KM*ZD].reshape(-1,KM,ZD),out[:,KM+KM*ZD:KM+2*KM*ZD].reshape(-1,KM,ZD),out[:,-1]
    def nll(s,out,zn,d,w):
        """返回损失和 dL/dout。w 是每样本的掩码"""
        pi,mu,ls,dl=s.split(out);ls=np.clip(ls,-5,2);lp=pi-pi.max(1,keepdims=True);lw=lp-np.log(np.exp(lp).sum(1,keepdims=True));e=(zn[:,None]-mu)/np.exp(ls);lg=lw-.5*(e*e+2*ls).sum(2);m=lg.max(1,keepdims=True);res=np.exp(lg-m);ss=res.sum(1,keepdims=True);L=-(m+np.log(ss))[:,0];R=res/ss;piw=np.exp(lw);Gpi=(piw-R);Gmu=-R[:,:,None]*e/np.exp(ls);Gls=R[:,:,None]*(1-e*e);p=sig(dl);nd=(5*d+(1-d));Ld=-nd*(d*np.log(p+1e-9)+(1-d)*np.log(1-p+1e-9));Gd=nd*(p-d);G=np.c_[Gpi,Gmu.reshape(len(out),-1),Gls.reshape(len(out),-1),Gd];return ((L+Ld)*w).sum(),G*w[:,None]
    def loss_grad(s,Z,A,D,Ma):
        """Z:(B,T,ZD) A,D,Ma:(B,T)。预测 z[t+1], d[t+1] 由 z[t],a[t]"""
        B,Tn=A.shape;h=np.zeros((B,RN));c=np.zeros((B,RN));C=[];Hs=[];tot=0.;Gout=[]
        for t in range(Tn-1):
            h,c,k=s.step(Z[:,t],A[:,t],h,c);C.append(k);Hs.append(h);out=h@s.O+s.ob;v,g=s.nll(out,Z[:,t+1],D[:,t+1],Ma[:,t+1]);tot+=v;Gout.append(g)
        N=Ma[:,1:].sum();G=[np.zeros_like(p) for p in s.P];dh=np.zeros((B,RN));dc=np.zeros((B,RN));n=RN
        for t in range(Tn-2,-1,-1):
            g=Gout[t];G[2]+=Hs[t].T@g;G[3]+=g.sum(0);dh=dh+g@s.O.T;x,f,i,oo,g2,Cc,tt,cp=C[t];dC=dc+dh*oo*(1-tt*tt);da=np.c_[dC*cp*f*(1-f),dC*g2*i*(1-i),dh*tt*oo*(1-oo),dC*i*(1-g2*g2)];dz=da@s.W.T;G[0]+=x.T@da;G[1]+=da.sum(0);dh=dz[:,ZD+3:];dc=dC*f
        return tot/N,[g/N for g in G]
    def upd(s,G,lr):
        s.n+=1;nm=np.sqrt(sum((g*g).sum() for g in G));G=[g*min(1,1/nm) for g in G]
        for k,(p,g) in enumerate(zip(s.P,G)):
            s.Mo[k]=.9*s.Mo[k]+.1*g;s.U[k]=.999*s.U[k]+.001*g*g;p-=lr*(s.Mo[k]/(1-.9**s.n))/(np.sqrt(s.U[k]/(1-.999**s.n))+1e-8)
    def sample_next(s,h,r,tau):
        out=h@s.O+s.ob;pi,mu,ls,dl=s.split(out);ls=np.clip(ls,-5,2);lp=pi/tau;lp=lp-lp.max(1,keepdims=True);q=np.exp(lp);q/=q.sum(1,keepdims=True);k=(q.cumsum(1)>r.random((len(h),1))).argmax(1);i=np.arange(len(h));return mu[i,k]+np.exp(ls[i,k])*np.sqrt(tau)*r.normal(0,1,(len(h),ZD)),sig(dl)>.5
def act(theta,fe):
    """线性控制器：连续输出分成三段（论文附录 A.3）"""
    a=np.tanh(fe@theta[:-1]+theta[-1]);return np.digitize(a,[-1/3,1/3])
def rollouts(net,V,theta_b,seed,tau=None,zonly=False,maxt=MAXT,reps=1):
    """同时评估多组控制器。tau=None 在真环境；否则在梦里。"""
    P,R=theta_b.shape[0],reps;n=P*R;r=np.random.default_rng(seed);h=np.zeros((n,RN));c=np.zeros((n,RN));alive=np.ones(n,bool);sc=np.zeros(n);th=np.repeat(theta_b,R,0)
    if tau is None:
        env=Env(n,r);z=V.enc(env.obs())[0]
    else:
        z=DREAM0[r.integers(0,len(DREAM0),n)]
    for t in range(maxt):
        fe=z if zonly else np.c_[z,h];a=(np.tanh((fe*th[:,:fe.shape[1]]).sum(1)+th[:,-1]));ai=np.digitize(a,[-1/3,1/3]);h,c,_=net.step(z,ai,h,c)
        sc+=alive
        if tau is None:
            hit=env.step(ai);alive=alive&~hit;z=V.enc(env.obs())[0]
        else:
            z,d=net.sample_next(h,r,tau);alive=alive&~d
        if not alive.any():break
    return sc.reshape(P,R).mean(1),sc
def cem(fit,dim,gens,pop,seed,sigma=.5):
    r=np.random.default_rng(seed);m=np.zeros(dim);s=np.full(dim,sigma);h=[];best=None;bf=-1e9
    for g in range(gens):
        T=m+s*r.normal(0,1,(pop,dim));f=fit(T,seed*100+g);el=T[np.argsort(-f)[:pop//4]];m=el.mean(0);s=el.std(0)+.02;h.append((round(float(f.mean()),1),round(float(f.max()),1)))
    return m,h
DREAM0=None
if __name__=='__main__':
    sd=int(sys.argv[1]);G=int(sys.argv[2]) if len(sys.argv)>2 else 40;out={};t0=time.time();r=np.random.default_rng(sd+1)
    if G<=0:
        m=M(0);Z=r.normal(0,1,(3,5,ZD));A=r.integers(0,3,(3,5));D=np.zeros((3,5));Ma=np.ones((3,5));L,Gr=m.loss_grad(Z,A,D,Ma)
        for k,nm in enumerate(('W','b','O','ob')):
            i=np.unravel_index(int(np.argmax(abs(Gr[k]))),Gr[k].shape);e=1e-5;m.P[k][i]+=e;a=m.loss_grad(Z,A,D,Ma)[0];m.P[k][i]-=2*e;b=m.loss_grad(Z,A,D,Ma)[0];m.P[k][i]+=e;print(nm,Gr[k][i],(a-b)/2/e)
        sys.exit()
    # ① 真环境随机回合
    N=800;env=Env(N,r);O=[];Ac=[];Do=[];Ma=[];Tm=120
    for t in range(Tm):
        a=r.integers(0,3,N);O.append(env.obs());Ac.append(a);Ma.append(env.alive.copy());hit=env.step(a);Do.append(env.alive.copy()==False)
        env.alive=env.alive|False
    O=np.stack(O,1);Ac=np.stack(Ac,1);Ma=np.stack(Ma,1);Dd=np.stack(Do,1);rnd=float(Ma.sum(1).mean());out['random']=rnd;print(sd,'随机策略存活',rnd,flush=True)
    X=O[Ma];V=VAE(sd);out['vae_err']=V.train(X,r,6000);print(sd,'VAE 错像素',out['vae_err'],len(X),round(time.time()-t0),flush=True)
    mu,lv=V.enc(O.reshape(-1,DIM));mu=mu.reshape(N,Tm,ZD);lv=lv.reshape(N,Tm,ZD);DREAM0=mu[:,0]
    net=M(sd);Lh=[]
    for st in range(1,3001):
        k=r.integers(0,N,32);s0=(r.random(32)*np.maximum(Ma[k].sum(1)-1,1)).astype(int);Kk=k[:,None];Tt_=np.minimum(s0[:,None]+np.arange(40),Tm-1);Z=mu[Kk,Tt_]+np.exp(.5*lv[Kk,Tt_])*r.normal(0,1,(32,40,ZD));Mm=Ma[Kk,Tt_]&(s0[:,None]+np.arange(40)<Tm);L,Gr=net.loss_grad(Z,Ac[Kk,Tt_],(Dd[Kk,Tt_]&Mm)*1.,Mm*1.);assert np.isfinite(L),'M 损失出现 NaN';net.upd(Gr,1e-3*(1 if st<2400 else .2))
        if st%500==0:Lh.append((st,round(float(L),2)))
    out['m_hist']=Lh;print(sd,'M',Lh,round(time.time()-t0),flush=True)
    D_full=ZD+RN+1;D_z=ZD+1;out['dream']={}
    for tau in (.1,.5,1.,1.2,1.5):
        def fit(T,s_):return rollouts(net,V,T,s_,tau=tau,reps=6)[0]
        th,h=cem(fit,D_full,G,32,sd+int(tau*100));ve=rollouts(net,V,th[None],5,tau=tau,reps=100)[1];ae=rollouts(net,V,th[None],6,reps=100)[1];out['dream'][str(tau)]=dict(virtual=[float(ve.mean()),float(ve.std())],actual=[float(ae.mean()),float(ae.std())],hist=h);print(sd,'τ',tau,out['dream'][str(tau)]['virtual'],out['dream'][str(tau)]['actual'],round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
        if sd==0 and tau==1.2:np.savez('m.npz',d_W0=V.D.W[0],d_W1=V.D.W[1],d_b0=V.D.b[0],d_b1=V.D.b[1],e_W0=V.E.W[0],e_W1=V.E.W[1],e_b0=V.E.b[0],e_b1=V.E.b[1],W=net.W,b=net.b,O=net.O,ob=net.ob,th=th,z0=DREAM0[:40])
    # ② 真环境训练（Car Racing 那种）：只给 z 和 z+h
    out['real']={}
    for k,zo,dm in (('z',True,D_z),('zh',False,D_full)):
        n=[0]
        def fit(T,s_):
            f,_=rollouts(net,V,(np.c_[T[:,:-1],np.zeros((len(T),D_full-dm)),T[:,-1]] if zo else T),s_,zonly=zo,reps=6);return f
        th,h=cem(fit,dm,G,32,sd+500);Tt=(np.r_[th[:-1],np.zeros(D_full-dm),th[-1]] if zo else th);ae=rollouts(net,V,Tt[None],6,zonly=zo,reps=100)[1];out['real'][k]=dict(actual=[float(ae.mean()),float(ae.std())],hist=h,rollouts=32*6*G);print(sd,'真环境',k,out['real'][k]['actual'],round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
