#!/usr/bin/env python3
# 第 40 期 π0 玩具实验：流匹配、跨机器人预训练和微调
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

"""第 40 期玩具实验：流匹配 + 跨机器人预训练 + 微调。平面上绕开中间的圆盘走到目标，可以上绕也可以下绕（两种模式）。
几个“机器人”关节数不同（2–3 维，零填充到 3 维），关节到末端是线性映射。语言指令是目标编号（one-hot）。
预训练数据多样但不干净，带纠偏；微调数据少而干净。MLP 代替 Transformer，没有图像，也没有视觉语言模型预训练。这是玩具。"""
import sys,json,time,numpy as np
H=8;Q=3;R=6;G=12;NT=8;T=24;EX=6;W=128
OB=np.array([0.,0.]);RAD=.35
GOALS=np.array([[.65+.3*(i%3)/2,-.66+1.32*i/(G-1)] for i in range(G)])
def robots():
    r=np.random.default_rng(123);Ms=[]
    for i,d in enumerate((3,3,2,2,3,3)):
        m=r.normal(0,1,(2,d));u,s,v=np.linalg.svd(m,full_matrices=False);m=u@np.diag(np.linspace(1.3,.8,2))@v;Ms.append(np.c_[m,np.zeros((2,Q-d))])
    return Ms
MS=robots();PI=[np.linalg.pinv(m) for m in MS]
def path(r,g,clean):
    """起点到目标的一条路，上绕或下绕。不干净的数据有抖动、速度变化和偏离后的纠偏"""
    s=np.array([-1.,0.])+r.normal(0,.03 if clean else .12,2);side=1. if r.random()<.5 else -1.;n=T
    wy=side*(.62+(0 if clean else r.normal(0,.06)));wp=np.array([0.,wy]);u=np.linspace(0,1,n)
    if not clean:
        v=np.cumsum(np.maximum(.3,1+r.normal(0,.5,n)));u=(v-v[0])/(v[-1]-v[0])
    p=((1-u)**2)[:,None]*s+(2*u*(1-u))[:,None]*(2*wp-.5*(s+g))+(u**2)[:,None]*g
    if not clean:
        if r.random()<.4:
            k=int(r.integers(0,n-14));off=r.normal(0,.2,2);p[k:]+=off*np.exp(-np.arange(n-k)/4.)[:,None]
        p=p+r.normal(0,.008,p.shape)
    return p
def demos(rb,goals,N,clean,seed):
    r=np.random.default_rng(seed);O=[];A=[]
    for _ in range(N):
        g=int(goals[r.integers(0,len(goals))]);q=path(r,GOALS[g],clean)@PI[rb].T
        for t in range(T-1):
            idx=np.minimum(np.arange(t+1,t+1+H),T-1);o=np.zeros(Q+R+G);o[:Q]=q[t];o[Q+rb]=1;o[Q+R+g]=1;O.append(o);A.append(q[idx].reshape(-1))
    return np.array(O,np.float32),np.array(A,np.float32)
class MLP:
    def __init__(s,di,H_,do,L,seed,last=1e-2):
        r=np.random.default_rng(seed);s.W=[r.normal(0,1/np.sqrt(di),(di,H_))]+[r.normal(0,1/np.sqrt(H_),(H_,H_)) for _ in range(L-1)]+[r.normal(0,last,(H_,do))];s.b=[np.zeros(H_) for _ in range(L)]+[np.zeros(do)];s.P=s.W+s.b;s.M=[np.zeros_like(p) for p in s.P];s.U=[np.zeros_like(p) for p in s.P];s.n=0;s.L=L
    def fw(s,x):
        hs=[x];h=x
        for i in range(s.L):
            n=np.maximum(h@s.W[i]+s.b[i],0)
            if i>0:n=n+h
            hs.append(n);h=n
        return h@s.W[-1]+s.b[-1],hs
    def grads(s,hs,g):
        B=len(g);G_=[None]*(2*(s.L+1));G_[s.L]=hs[s.L].T@g/B;G_[2*s.L+1]=g.mean(0);u=g@s.W[-1].T
        for i in range(s.L-1,-1,-1):
            pre=hs[i];act=(pre@s.W[i]+s.b[i])>0;d=u*act;G_[i]=pre.T@d/B;G_[s.L+1+i]=d.mean(0);u=d@s.W[i].T+(u if i>0 else 0)
        return u,G_
    def upd(s,G_,lr):
        s.n+=1
        for k,(p,g) in enumerate(zip(s.P,G_)):
            s.M[k]=.9*s.M[k]+.1*g;s.U[k]=.999*s.U[k]+.001*g*g;p-=lr*(s.M[k]/(1-.9**s.n))/(np.sqrt(s.U[k]/(1-.999**s.n))+1e-8)
    def clone(s,seed=0):
        m=MLP(s.W[0].shape[0],W,s.W[-1].shape[1],s.L,seed);m.W=[w.copy() for w in s.W];m.b=[b.copy() for b in s.b];m.P=m.W+m.b;return m
def tf(tau):
    return np.c_[tau,np.sin(np.pi*tau),np.cos(np.pi*tau),np.sin(2*np.pi*tau),np.cos(2*np.pi*tau),np.sin(4*np.pi*tau),np.cos(4*np.pi*tau),np.sin(8*np.pi*tau)]
DA=H*Q;DO=Q+R+G
def new(flow,seed):
    return MLP(DO+(DA+NT if flow else 0),W,DA,3,seed)
def sample_tau(r,n,mode):
    if mode=='uniform':return r.random(n)
    s=.999;return s*(1-r.beta(1.5,1,n))
def train(m,O,A,steps,seed,flow,lr0=1e-3,pw=None,tau='beta',bs=128):
    r=np.random.default_rng(seed)
    for st in range(1,steps+1):
        i=r.integers(0,len(O),bs) if pw is None else r.choice(len(O),bs,p=pw);o,a=O[i],A[i];lr=lr0*(1 if st<.8*steps else .2)
        if flow:
            t=sample_tau(r,bs,tau);e=r.normal(0,1,a.shape);x=t[:,None]*a+(1-t[:,None])*e;y,hs=m.fw(np.c_[o,x,tf(t)]);tgt=a-e
        else:
            y,hs=m.fw(o);tgt=a
        _,Gs=m.grads(hs,2*(y-tgt));m.upd(Gs,lr)
    return m
def act(m,o,flow,r,steps=10,sign=1.):
    if not flow:return m.fw(o)[0]
    x=r.normal(0,1,(len(o),DA));d=1./steps
    for i in range(steps):
        t=np.full(len(o),i*d);x=x+sign*d*m.fw(np.c_[o,x,tf(t)])[0]
    return x
def rollout(m,rb,g,flow,seed,n=80,steps=10,sign=1.,kick=False,noise=.01,gain=.7):
    r=np.random.default_rng(seed);x=np.array([-1.,0.])+r.normal(0,.03,(n,2));q=x@PI[rb].T;hit=np.zeros(n,bool);o=np.zeros((n,DO));o[:,Q+rb]=1;o[:,Q+R+g]=1;up=np.zeros(n)
    for t in range(0,T,EX):
        o[:,:Q]=q;A=act(m,o,flow,r,steps,sign).reshape(n,H,Q)
        for j in range(EX):
            q=q+gain*(A[:,j]-q)+r.normal(0,noise,q.shape)*(np.abs(PI[rb]).sum(1)>0);ee=q@MS[rb].T
            if kick and t+j==6:
                d=r.normal(0,.25,(n,2));q=q+d@PI[rb].T;ee=q@MS[rb].T
            hit|=np.linalg.norm(ee-OB,axis=1)<RAD-.05;up=np.maximum(up,np.where(np.abs(ee[:,0])<.1,np.sign(ee[:,1]),0))
    ok=(np.linalg.norm(ee-GOALS[g],axis=1)<.2)&~hit
    return dict(succ=float(ok.mean()),hit=float(hit.mean()),reach=float((np.linalg.norm(ee-GOALS[g],axis=1)<.2).mean()),up=float((up>0).mean()))
def ev(m,rb,goals,flow,seed,**k):
    rs=[rollout(m,rb,int(g),flow,seed+i,n=40,**k) for i,g in enumerate(goals)];return {f:float(np.mean([x[f] for x in rs])) for f in rs[0]}
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 6000;F=int(sys.argv[3]) if len(sys.argv)>3 else 1000;t0=time.time();res={}
    if S<=0:
        m=MLP(10,16,4,3,0);r=np.random.default_rng(0);x=r.normal(0,1,(4,10));y=r.normal(0,1,(4,4));o,hs=m.fw(x);u,Gs=m.grads(hs,2*(o-y));L=lambda:float(((m.fw(x)[0]-y)**2).sum(1).mean())
        for k,q in enumerate(m.P):
            i=np.unravel_index(int(np.argmax(abs(Gs[k]))),q.shape);e=1e-6;q[i]+=e;a=L();q[i]-=2*e;b=L();q[i]+=e;print(k,round(float(Gs[k][i]),6),round((a-b)/2/e,6))
        sys.exit()
    # 预训练任务：每个机器人只见 8 个目标；新机器人 5 不参加。样本数不平衡（0 号最多）
    rg=np.random.default_rng(777);seen=[np.sort(rg.choice(G,8,replace=False)) for _ in range(R)]
    sizes=(1200,400,150,50,20);Dt=[demos(i,seen[i],n,False,sd*10+i) for i,n in enumerate(sizes)]
    Op=np.concatenate([d[0] for d in Dt]);Ap=np.concatenate([d[1] for d in Dt]);cnt=np.concatenate([np.full(len(d[0]),len(d[0])) for d in Dt]);pw=cnt**(.43-1.);pw=pw/pw.sum()
    res['seen']=[s.tolist() for s in seen];res['samples']=[len(d[0]) for d in Dt]
    bases={}
    for flow in (True,False):
        bases[flow]=train(new(flow,sd),Op,Ap,S,sd+1,flow,pw=pw);print(sd,'pre',flow,round(time.time()-t0),flush=True)
    res['oob']={('flow' if f else 'mse'):[ev(bases[f],i,seen[i],f,500+sd) for i in range(5)] for f in (True,False)};print(res['oob']['flow'],flush=True)
    # 采样比例：n^0.43 与自然比例
    m=train(new(True,sd),Op,Ap,S,sd+1,True,pw=None);res['natural']=[ev(m,i,seen[i],True,500+sd) for i in range(5)];print('nat',res['natural'],flush=True)
    # tau 均匀采样
    m=train(new(True,sd),Op,Ap,S,sd+1,True,pw=pw,tau='uniform');res['tau_uniform']=ev(m,0,seen[0],True,500+sd);res['tau_beta']=res['oob']['flow'][0]
    # Euler 步数、符号
    res['euler']={str(k):ev(bases[True],0,seen[0],True,500+sd,steps=k) for k in (1,2,5,10,50)}
    res['sign_literal']=ev(bases[True],0,seen[0],True,500+sd,sign=-1.);print('euler',res['euler'],res['sign_literal'],flush=True)
    # 微调：易（1 号机器人，它没见过的目标）；难（新机器人 5，它的目标都没见过）
    rest=[int(g) for g in range(G) if g not in seen[1]]
    tiers={'easy':(1,rest),'hard':(5,list(range(G))[::3])}
    for tn,(rb,gs) in tiers.items():
        res[tn]={}
        for N in (3,10,40):
            Of,Af=demos(rb,gs,N,True,sd+900+N);row={}
            for flow,fn in ((True,'flow'),(False,'mse')):
                ms=train(new(flow,sd+2),Of,Af,F,sd+3,flow);mf=train(bases[flow].clone(),Of,Af,F,sd+3,flow,lr0=3e-4)
                for nm,mm in (('scratch',ms),('fine',mf)):
                    row[fn+'_'+nm]=ev(mm,rb,gs,flow,700+sd);row[fn+'_'+nm+'_kick']=ev(mm,rb,gs,flow,700+sd,kick=True)
                if sd==0 and tn=='easy' and N==10:np.savez(f'm_{fn}.npz',**{f'W{i}':w for i,w in enumerate(mf.W)},**{f'b{i}':w for i,w in enumerate(mf.b)})
            res[tn][str(N)]=row;print(sd,tn,N,{k:round(v['succ'],2) for k,v in row.items()},round(time.time()-t0),flush=True);json.dump(res,open(f'res{sd}.json','w'))
        res[tn]['oob']={fn:ev(bases[f],rb,gs,f,700+sd) for f,fn in ((True,'flow'),(False,'mse'))}
    res['rest']=rest;res['M1']=MS[1].tolist();res['PI1']=PI[1].tolist();res['sec']=time.time()-t0;json.dump(res,open(f'res{sd}.json','w'));print('done',round(time.time()-t0),flush=True)
