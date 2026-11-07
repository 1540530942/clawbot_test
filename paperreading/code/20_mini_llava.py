#!/usr/bin/env python3
# 第 20 期 LLaVA（需要第 10 期的 clip.npz）
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

"""第 20 期玩具实验：迷你 LLaVA。视觉编码器用第 10 期训练好的迷你 CLIP（冻结，需要 clip.npz），
语言模型先只读文字学会问答（相当于 Vicuna），第一阶段只训练一个线性投影，第二阶段连语言模型一起微调。
问答数据由程序从标准描述生成（相当于论文里只看文字的 GPT-4 老师）。这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
SH=['横线','竖线','方框','十字','对角线','小点'];PO=['左上','右上','左下','右下']
TPL=[lambda s,p:[p,'的',s],lambda s,p:[s,'在',p],lambda s,p:['一张',p,'的',s,'的','图']]
vocab=['<p>','<e>']+SH+PO+['的','在','一张','图'];ID={w:i for i,w in enumerate(vocab)};V=len(vocab);LT=8;H=16;P=4;NP=16
HOLD={(0,3),(1,0),(2,2),(3,1),(4,3),(5,0)}   # (形状,位置) 训练时从不出现的组合
def draw(r,c,q):
    a=np.zeros((H,H));s=r.integers(4,7);oy=(q//2)*8;ox=(q%2)*8;x=ox+r.integers(0,8-s+1);y=oy+r.integers(0,8-s+1)
    if c==0:a[y+s//2,x:x+s]=1
    elif c==1:a[y:y+s,x+s//2]=1
    elif c==2:a[y,x:x+s]=1;a[y+s-1,x:x+s]=1;a[y:y+s,x]=1;a[y:y+s,x+s-1]=1
    elif c==3:a[y+s//2,x:x+s]=1;a[y:y+s,x+s//2]=1
    elif c==4:
        for i in range(s):a[y+i,x+i]=1
    else:a[y+s//2:y+s//2+2,x+s//2:x+s//2+2]=1
    return np.clip(a+r.normal(0,.1,a.shape),0,1)
def enc_text(ws):return [ID[w] for w in ws+['<e>']]+[0]*(LT-len(ws)-1)
def patches(X):
    B=len(X);return X.reshape(B,4,4,4,4).transpose(0,1,3,2,4).reshape(B,NP,16)
def l2(x):
    n=np.sqrt((x.v**2).sum(-1,keepdims=True)+1e-9);r=T(x.v/n,(x,));u=x.v/n
    r.bk=lambda g:((g-(g*u).sum(-1,keepdims=True)*u)/n,);return r
def scale(s,t):
    k=np.exp(t.v[0]);r=T(s.v*k,(s,t));r.bk=lambda g:(g*k,np.array([(g*s.v*k).sum()]));return r
class Blocks:
    def __init__(s,o,d,h,ff,N,r):
        lin=lambda a,b:o.par(r.normal(0,1/np.sqrt(a),(a,b)));s.d,s.h=d,h
        s.L=[dict(q=lin(d,d),k=lin(d,d),v=lin(d,d),o=lin(d,d),a=lin(d,ff),ab=o.par(np.zeros(ff)),b=lin(ff,d),bb=o.par(np.zeros(d))) for _ in range(N)]
    def attn(s,l,x):
        B,n,_=x.v.shape;h,dk=s.h,s.d//s.h;sp=lambda t:t.rs(B,n,h,dk).t(0,2,1,3);Q,K,W=sp(x@l['q']),sp(x@l['k']),sp(x@l['v'])
        return ((Q@K.t(0,1,3,2)*(1/np.sqrt(dk))).softmax()@W).t(0,2,1,3).rs(B,n,h*dk)@l['o']
    def __call__(s,x):
        for l in s.L:
            x=x+s.attn(l,x.lnorm());g=x.lnorm();x=x+(((g@l['a'])+l['ab']).relu()@l['b'])+l['bb']
        return x.lnorm()
class CLIP:
    def __init__(s,seed=0,d=32,de=24):
        r=np.random.default_rng(seed);s.ps=[];s.d=d;lin=lambda a,b:s.par(r.normal(0,1/np.sqrt(a),(a,b)))
        s.PE=lin(16,d);s.CL=s.par(r.normal(0,.1,(1,d)));s.PI=s.par(r.normal(0,.1,(NP+1,d)));s.VB=Blocks(s,d,4,64,2,r);s.VP=lin(d,de)
        s.TE=lin(V,d);s.PT=s.par(r.normal(0,.1,(LT,d)));s.TB=Blocks(s,d,4,64,2,r);s.TP=lin(d,de);s.logT=s.par(np.array([np.log(1/.07)]))
    def par(s,v):t=T(v);s.ps.append(t);return t
    def image(s,X):
        B=len(X);p=T(patches(X))@s.PE;cls=T(np.ones((B,1,1)))@s.CL;A=np.zeros((1,NP+1));A[0,0]=1
        x=T(np.tile(np.eye(NP+1)[:,1:],(B,1,1)))@p+T(np.tile(A.T,(B,1,1)))@cls+s.PI
        h=s.VB(x);pick=np.zeros((B,NP+1,1));pick[:,0,0]=1
        return l2((h.t(0,2,1)@T(pick)).rs(B,s.d)@s.VP)
    def text(s,ids):
        ids=np.array(ids);B=len(ids);x=T(np.eye(V)[ids])@s.TE+T(np.eye(LT))@s.PT;h=s.TB(x);pick=np.zeros((B,LT,1))
        pick[np.arange(B),(ids==1).argmax(1),0]=1
        return l2((h.t(0,2,1)@T(pick)).rs(B,s.d)@s.TP)

# ---- 下面是语言部分 ----
W_=['<p>','<e>']+SH+PO+['文','图','问','答',';','形状','位置','描述','吗','是','否','对角','的']
ID2={w:i for i,w in enumerate(W_)};V2=len(W_);NV=NP+1;L2=NV+22;PAD=0
OPP={'左上':'右下','右上':'左下','左下':'右上','右下':'左上'}
def qa(r,c,q,kinds):
    s,p=SH[c],PO[q];k=kinds[int(r.integers(len(kinds)))]
    if k=='shape':return ['形状'],[s],k
    if k=='pos':return ['位置'],[p],k
    if k=='yes':
        t=s if r.random()<.5 else SH[int(r.integers(6))];return [t,'吗'],['是' if t==s else '否'],k
    if k=='detail':return ['描述'],[p,'的',s],k
    return ['对角'],[OPP[p]],k   # 需要推理的一类
ALL=['shape','pos','yes','detail','opp'];CONV=['shape','pos','yes']
def pick(r,hold=False):
    while True:
        c,q=int(r.integers(6)),int(r.integers(4))
        if ((c,q) in HOLD)==hold:return c,q
def tok(ws,n):return [ID2[w] for w in ws]+[PAD]*(n-len(ws))
class LM(Enc):
    def __init__(s,seed):
        super().__init__(V2,L2,d=32,h=4,ff=64,N=2,seed=seed,causal=True)
    def run(s,h):
        for l in s.layers:
            h=(h+s.attn(l,h)).lnorm();h=(h+(((h@l['a'])+l['ab']).relu()@l['b'])+l['bb']).lnorm()
        return h@s.E.t(1,0)
    def text(s,x):
        n=x.shape[1];return s.run(T(np.eye(s.V)[x])@s.E*np.sqrt(s.d)+T(np.eye(s.L)[:n])@s.P)
    def vis(s,Z,x,Wp):
        """前 NV 个位置放投影后的视觉向量，后面是词"""
        B,n=x.shape;M=np.zeros((n+NV,NV));M[:NV,:NV]=np.eye(NV);one=np.zeros((B,n+NV,s.V));one[:,NV:,:]=np.eye(s.V)[x]
        return s.run(T(np.tile(M,(B,1,1)))@(T(Z)@Wp)+T(one)@s.E*np.sqrt(s.d)+T(np.eye(s.L)[:n+NV])@s.P)
def feat(C,X,mode='last'):
    B=len(X);p=T(patches(X))@C.PE;A=np.zeros((1,NP+1));A[0,0]=1
    x=T(np.tile(np.eye(NP+1)[:,1:],(B,1,1)))@p+T(np.tile(A.T,(B,1,1)))@(T(np.ones((B,1,1)))@C.CL)+C.PI
    L=C.VB.L
    for i,l in enumerate(L):
        if mode=='before' and i==len(L)-1:break
        x=x+C.VB.attn(l,x.lnorm());g=x.lnorm();x=x+(((g@l['a'])+l['ab']).relu()@l['b'])+l['bb']
    return x.lnorm().v
def textbatch(r,B):
    S=[]
    for _ in range(B):
        c,q=pick(r,bool(r.random()<.2));Q,A,_=qa(r,c,q,ALL);S.append((['文',PO[q],'的',SH[c],';','问']+Q+['答'],A+['<e>']))
    n=max(len(a)+len(b) for a,b in S);y=np.array([tok(a+b,n) for a,b in S]);w=np.zeros((B,n-1))
    for i,(a,b) in enumerate(S):w[i,len(a)-1:len(a)+len(b)-1]=1
    return y[:,:-1],y[:,1:],w
def visbatch(r,B,C,mode,kinds,caption=False):
    X=[];S=[]
    for _ in range(B):
        c,q=pick(r);X.append(np.clip(draw(r,c,q),0,1))
        if caption:S.append((['图','描述','答'],[PO[q],'的',SH[c],'<e>']))
        else:
            Q,A,_=qa(r,c,q,kinds);S.append((['图','问']+Q+['答'],A+['<e>']))
    n=max(len(a)+len(b) for a,b in S);y=np.array([tok(a+b,n) for a,b in S]);Bx,Ln=y.shape;w=np.zeros((Bx,NV+Ln-1));t=np.zeros((Bx,NV+Ln-1),int)
    for i,(a,b) in enumerate(S):
        t[i,NV:]=y[i,1:];w[i,NV+len(a)-1:NV+len(a)+len(b)-1]=1
    return feat(C,np.array(X),mode),y[:,:-1],t,w
def clone(w,seed=0):
    m=LM(seed)
    for p,v in zip(m.ps,w):p.v=v.copy()
    return m
def stage(m,Wp,r,C,mode,steps,params,kinds,caption,lr):
    o=Adam(params)
    for st in range(1,steps+1):
        Z,x,t,w=visbatch(r,32,C,mode,kinds,caption);v,p,g=ce(m.vis(Z,x,Wp),t,w);p.backward(g)
        for q in params:
            if q.g is None:q.g=np.zeros_like(q.v)
        keep=set(map(id,params))
        for q in m.ps+[Wp]:
            if id(q) not in keep:q.g=None
        o.step(lr*min(1,st/50)*(.3 if st>.8*steps else 1))
def answer(m,Wp,Z,Q,n=4):
    x=np.array([tok(['图','问']+Q+['答'],3+len(Q))])
    o=[];ids=list(x[0])
    for _ in range(n):
        lg=m.vis(Z[None],np.array([ids]),Wp).v[0,-1];lg[PAD]=-1e9;k=int(lg.argmax())
        if W_[k]=='<e>':break
        o.append(W_[k]);ids.append(k)
    return o
def evaluate(m,Wp,C,mode,seed=5,n=120):
    r=np.random.default_rng(seed);out={}
    for hold in(False,True):
        for k in ALL:
            ok=0
            for _ in range(n):
                c,q=pick(r,hold);Z=feat(C,np.clip(draw(r,c,q),0,1)[None],mode)[0];Q,A,_=qa(r,c,q,[k]);ok+=answer(m,Wp,Z,Q)==A
            out[('hold_' if hold else 'seen_')+k]=ok/n
    return out
if __name__=='__main__':
    sd=int(sys.argv[1]);a=int(sys.argv[2]) if len(sys.argv)>2 else 2500;b=int(sys.argv[3]) if len(sys.argv)>3 else 800
    C=CLIP(seed=0);d=np.load('clip.npz')
    for i,p in enumerate(C.ps):p.v=d[f'p{i}']
    r=np.random.default_rng(sd+3);t0=time.time()
    M=LM(sd);o=Adam(M.ps)
    for st in range(1,a+1):
        x,y,w=textbatch(r,32);v,p,g=ce(M.text(x),y,w);p.backward(g);o.step(2e-3*min(1,st/100)*(.3 if st>.8*a else 1))
    Base=[p.v.copy() for p in M.ps];print('text lm',round(float(v),3),round(time.time()-t0),flush=True)
    newW=lambda s:T(np.random.default_rng(s).normal(0,1/np.sqrt(32),(32,32)))
    out={'params_W':32*32,'params_lm':int(sum(x.size for x in Base))}
    def run(name,mode='last',stage1=True,kinds=ALL,tune=True):
        m=clone(Base);Wp=newW(sd+1)
        if stage1:stage(m,Wp,r,C,mode,b,[Wp],None,True,2e-3)
        if tune:stage(m,Wp,r,C,mode,b,m.ps+[Wp],kinds,False,6e-4)
        out[name]=evaluate(m,Wp,C,mode);print(sd,name,{k:round(v,2) for k,v in out[name].items()},round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
        return m,Wp
    run('只有第一阶段',tune=False)
    run('只微调对话',kinds=CONV)
    mf,Wf=run('完整')
    run('跳过第一阶段',stage1=False)
    run('用倒数第二层特征',mode='before')
    if sd==0:
        np.savez('llava.npz',W=Wf.v,**{f'p{i}':p.v for i,p in enumerate(mf.ps)})
