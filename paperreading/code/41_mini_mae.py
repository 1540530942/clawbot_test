#!/usr/bin/env python3
# 第 41 期 MAE 玩具实验：遮挡比例、线性探测、微调和旋钮对比
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

"""第 41 期玩具实验：迷你 MAE。16×16 的合成图，切成 16 个 4×4 小块。随机遮掉一部分，编码器只看可见块，小解码器猜回像素。
没有标签地预训练后，用少量标签做线性探测和微调。比较遮挡比例、编码器要不要掩码词元、只算遮挡块的损失、遮挡方式、归一化目标。这是玩具。"""
import sys,json,time,numpy as np
S=16;P=4;G=4;N=16;K=6;NAMES=['方块','圆','三角','十字','圆环','L 形']
def shape(r,c):
    """一张图：一个主形状（决定类别），加噪声和小干扰点"""
    yy,xx=np.mgrid[:S,:S];s=int(r.integers(8,13));x0,y0=r.integers(0,S-s+1,2);cx,cy=x0+(s-1)/2,y0+(s-1)/2;R=s/2;u,v=(xx-cx)/R,(yy-cy)/R;b=(abs(u)<=1)&(abs(v)<=1)
    if c==0:m=b&(abs(u)<=.8)&(abs(v)<=.8)
    elif c==1:m=u*u+v*v<=.9
    elif c==2:m=b&(v>=2*abs(u)-1)
    elif c==3:m=b&((abs(u)<.35)|(abs(v)<.35))
    elif c==4:m=(u*u+v*v<=1.)&(u*u+v*v>=.35)
    else:m=b&((u<-.2)|(v>.2))
    a=r.uniform(.5,1)*r.choice([-1,1]);img=r.normal(0,.08,(S,S))+a*m
    for _ in range(int(r.integers(0,3))):
        i,j=r.integers(0,S-1,2);img[i:i+2,j:j+2]+=r.uniform(-.4,.4)
    return img
def dataset(n,seed):
    r=np.random.default_rng(seed);y=r.integers(0,K,n);return np.stack([shape(r,int(c)) for c in y]),y
def patch(x):
    return x.reshape(-1,G,P,G,P).transpose(0,1,3,2,4).reshape(-1,N,P*P)
def unpatch(p):
    return p.reshape(-1,G,G,P,P).transpose(0,1,3,2,4).reshape(-1,S,S)
def sincos(d):
    f=1/(10000**(np.arange(d//4)/(d//4)));yy,xx=np.mgrid[:G,:G];x=xx.ravel()[:,None]*f;y=yy.ravel()[:,None]*f;return np.c_[np.sin(x),np.cos(x),np.sin(y),np.cos(y)]
class MAE:
    def __init__(s,seed,d=48,dd=32,Ne=2,Nd=2,h=4):
        r=np.random.default_rng(seed);s.d,s.dd,s.h=d,dd,h;s.ps=[];s.nm=[]
        def L(a,b,nm,sc=1.):
            t=T(r.normal(0,sc/np.sqrt(a),(a,b)));s.ps.append(t);s.nm.append(nm);return t
        s.In=L(P*P,d,'In');s.Pe=sincos(d);s.Pd=sincos(dd);s.Me=L(1,d,'Me',.1);s.Md=L(1,dd,'Md',.1);s.Pj=L(d,dd,'Pj')
        def blk(i,w,t):return {k:L(w,w,f'{t}{i}{k}') for k in 'qkvo'}|{'a':L(w,2*w,f'{t}{i}a'),'b':L(2*w,w,f'{t}{i}b')}
        s.E=[blk(i,d,'e') for i in range(Ne)];s.D=[blk(i,dd,'d') for i in range(Nd)];s.Out=L(dd,P*P,'Out');s.Cls=L(d,K,'Cls',.1)
    def attn(s,l,x):
        B,n,d=x.v.shape;dk=d//s.h;sp=lambda t:t.rs(B,n,s.h,dk).t(0,2,1,3);Q=sp(x@l['q']);K_=sp(x@l['k']);V=sp(x@l['v']);a=(Q@K_.t(0,1,3,2)*(1/np.sqrt(dk))).softmax();return (a@V).t(0,2,1,3).rs(B,n,s.h*dk)@l['o']
    def blocks(s,h,ls):
        for l in ls:
            g=h.lnorm();h=h+s.attn(l,g);g=h.lnorm();h=h+(g@l['a']).relu()@l['b']
        return h.lnorm()
    @staticmethod
    def fill(vis,mt,idx,d,pos):
        """把可见词元放回原位，其余位置填掩码词元。pos 是固定位置编码"""
        B=vis.v.shape[0];nv=idx.shape[1];v=np.tile(mt.v,(B,N,1));bi=np.arange(B)[:,None];v[bi,idx]=vis.v;mask=np.ones((B,N),bool);mask[bi,idx]=False;o=T(v,(vis,mt))
        def bk(g):
            return (g[bi,idx],g[mask].sum(0,keepdims=True))
        o.bk=bk;return o+pos
    def encode(s,p,idx=None,tok=False):
        B=len(p);e=T(p)@s.In
        if idx is None:return s.blocks(e+s.Pe,s.E)
        bi=np.arange(B)[:,None]
        if tok:
            vis=T(e.v[bi,idx],(e,));sh=e.v.shape
            def bk(g):
                o=np.zeros(sh);o[bi,idx]=g;return (o,)
            vis.bk=bk;return s.blocks(s.fill(vis,s.Me,idx,s.d,s.Pe),s.E)
        x=e+s.Pe;vis=T(x.v[bi,idx],(x,));sh=x.v.shape
        def bk2(g):
            o=np.zeros(sh);o[bi,idx]=g;return (o,)
        vis.bk=bk2;return s.blocks(vis,s.E)
    def decode(s,h,idx,tok=False):
        if tok:
            return s.blocks(h@s.Pj+s.Pd,s.D)@s.Out
        return s.blocks(s.fill(h@s.Pj,s.Md,idx,s.dd,s.Pd),s.D)@s.Out
    def feat(s,x):
        return s.encode(patch(x)).v.mean(1)
def masks(r,B,ratio,mode='random'):
    nv=max(1,int(round(N*(1-ratio))))
    if mode=='random':return np.argsort(r.random((B,N)),1)[:,:nv]
    if mode=='grid':
        o=[];
        for _ in range(B):
            a,b=r.integers(0,2,2);o.append([i*G+j for i in range(G) for j in range(G) if i%2==a and j%2==b])
        return np.array(o)
    if mode=='block':
        o=[];
        for _ in range(B):
            m=np.ones((G,G),bool);cnt=0
            while cnt<N-nv:
                w,h=r.integers(1,4,2);i,j=r.integers(0,G-h+1),r.integers(0,G-w+1);m[i:i+h,j:j+w]=False;cnt=int((~m).sum())
            v=np.flatnonzero(m.ravel());v=v if len(v)>0 else np.array([0]);o.append(np.resize(r.permutation(v),nv))
        return np.array(o)
def pretrain(X,steps,seed,ratio=.75,mode='random',tok=False,loss_all=False,norm=False,Nd=2,bs=64):
    m=MAE(seed,Nd=Nd);o=Adam(m.ps);r=np.random.default_rng(seed+5);P_=patch(X);t0=time.time();L=[]
    for st in range(steps):
        b=P_[r.integers(0,len(P_),bs)];idx=masks(r,bs,ratio,mode);h=m.encode(b,idx,tok);y=m.decode(h,idx,tok);t=b
        if norm:t=(b-b.mean(2,keepdims=True))/(b.std(2,keepdims=True)+1e-3)
        w=np.ones((bs,N));w[np.arange(bs)[:,None],idx]=0
        if loss_all:w=np.ones((bs,N))
        d=y.v-t;g=2*d*w[:,:,None]/(w.sum()*P*P);L.append(float(((d*d).mean(2)*w).sum()/w.sum()));y.backward(g);step(o,m,1e-3 if st<.8*steps else 2e-4)
    return m,time.time()-t0,float(np.mean(L[-100:]))
def step(o,m,lr):
    o.step(lr);[setattr(p,'g',None) for p in m.ps]
def probe(m,Xl,yl,Xt,yt,steps=600):
    """线性探测：特征标准化后做 softmax 回归"""
    A,B=m.feat(Xl),m.feat(Xt);mu,sd=A.mean(0),A.std(0)+1e-6;A,B=(A-mu)/sd,(B-mu)/sd;W=np.zeros((A.shape[1],K));Y=np.eye(K)[yl]
    for _ in range(steps):
        p=np.exp(A@W-(A@W).max(1,keepdims=True));p/=p.sum(1,keepdims=True);W-=.3*(A.T@(p-Y)/len(A)+1e-3*W)
    return float(((B@W).argmax(1)==yt).mean())
def tune(m,Xl,yl,Xt,yt,steps,seed,lr=5e-4,bs=32):
    """微调：全部权重一起训练。解码器不用"""
    ps=[m.In,*[v for l in m.E for v in l.values()],m.Cls];o=Adam(ps);r=np.random.default_rng(seed)
    for st in range(steps):
        i=r.integers(0,len(Xl),bs);h=m.encode(patch(Xl[i]));pool=T(h.v.mean(1),(h,));pool.bk=lambda g,n=N:(np.repeat(g[:,None,:],n,1)/n,);lg=pool@m.Cls;e=lg.v-lg.v.max(1,keepdims=True);p=np.exp(e);p/=p.sum(1,keepdims=True);g=(p-np.eye(K)[yl[i]])/bs;lg.backward(g);o.step(lr*(1 if st<.8*steps else .2))
        for q in m.ps:q.g=None
    return acc(m,Xt,yt)
def acc(m,X,y):
    c=0
    for i in range(0,len(X),128):c+=int((((m.encode(patch(X[i:i+128])).v.mean(1))@m.Cls.v).argmax(1)==y[i:i+128]).sum())
    return c/len(X)
if __name__=='__main__':
    sd=int(sys.argv[1]);Sd=int(sys.argv[2]) if len(sys.argv)>2 else 2500;Ft=int(sys.argv[3]) if len(sys.argv)>3 else 400;t0=time.time()
    if Sd<=0:
        m=MAE(0);r=np.random.default_rng(0);X,_=dataset(4,0);b=patch(X);idx=masks(r,4,.75);h=m.encode(b,idx);y=m.decode(h,idx);tg=r.normal(0,1,y.v.shape);y.backward(2*(y.v-tg))
        def L():
            return float(((m.decode(m.encode(b,idx),idx).v-tg)**2).sum())
        for p,n in zip(m.ps,m.nm):
            if n in('In','Md','Pj','e0q','d1a','Out'):
                i=np.unravel_index(int(np.argmax(abs(p.g))),p.v.shape);e=1e-6;p.v[i]+=e;a=L();p.v[i]-=2*e;c=L();p.v[i]+=e;print(n,round(float(p.g[i]),5),round((a-c)/2/e,5))
        sys.exit()
    Xu,_=dataset(6000,sd+1);Xl,yl=dataset(200,sd+2);Xt,yt=dataset(1000,sd+3);out={'names':NAMES}
    def run(tag,**k):
        m,sec,l=pretrain(Xu,Sd,sd,**k);pb=probe(m,Xl,yl,Xt,yt);ft50=tune(pickle(m),Xl[:50],yl[:50],Xt,yt,Ft,sd+7);ft=tune(pickle(m),Xl,yl,Xt,yt,Ft,sd+7);out[tag]=dict(sec=sec,loss=l,probe=pb,ft=ft,ft50=ft50);print(sd,tag,out[tag],round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False);return m
    def pickle(m):
        c=MAE(0)
        for a,b in zip(c.ps,m.ps):a.v=b.v.copy()
        return c
    s0=MAE(sd+50);s1=MAE(sd+50);out['scratch']=dict(probe=probe(s0,Xl,yl,Xt,yt),ft=tune(s0,Xl,yl,Xt,yt,Ft,sd+7),ft50=tune(s1,Xl[:50],yl[:50],Xt,yt,Ft,sd+7));print(out['scratch'],flush=True)
    s2=MAE(sd+50);out['scratch_long']=tune(s2,Xl,yl,Xt,yt,Ft*4,sd+7)
    for R in (.25,.5,.75,.9):
        m=run(f'r{int(R*100)}',ratio=R)
        if R==.75 and sd==0:np.savez('m_mae.npz',**{n:p.v for n,p in zip(m.nm,m.ps)},Pe=m.Pe,Pd=m.Pd)
    run('tok',tok=True);run('loss_all',loss_all=True);run('norm',norm=True);run('grid',mode='grid');run('block',mode='block');run('dec1',Nd=1)
    # 重建误差：用 75% 模型遮更多
    print('done',round(time.time()-t0),flush=True)
