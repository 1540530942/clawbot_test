#!/usr/bin/env python3
# 第 12 期迷你实验：预训练后适配到新任务，比较全量微调、前缀、适配器和不同秩的 LoRA。
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

"""第 12 期玩具实验：先预训练一个迷你 Transformer（倒序、加一、减一、两两交换），
再适配到新任务“倒序再加一”。比较从零训练、不适配、全量微调、只训偏置、只训最后一层、
前缀、适配器和不同秩的 LoRA。"""
import sys,json,time,numpy as np
PAD=10;TK=[11,12,13,14];V=15;N=10;L=32
def gen(r,B,task):
    d=r.integers(0,10,(B,N));x=np.c_[np.full((B,1),TK[min(task,3)]),d]
    if task==0: y=d[:,::-1]
    elif task==1: y=(d+1)%10
    elif task==2: y=(d-1)%10
    elif task==3: y=d.copy();y[:,0::2],y[:,1::2]=d[:,1::2],d[:,0::2]
    else: y=(d[:,::-1]+1)%10
    return x,np.c_[np.zeros((B,1),int),y],np.c_[np.zeros((B,1)),np.ones((B,N))]
DOWN=4
def downx(x): x=x.copy();x[:,0]=TK[0];return x   # 下游任务用同一个提示词，靠适配改变含义
class M(Enc):
    method='base';pf=None;lp=0;lora=();ad=()
    def setup(s,method,r=4,seed=0,which=('q','v'),lp=0):
        s.method,s.r=method,r;s.lora=[];s.ad=[];s.pf=None;rg=np.random.default_rng(1000+seed);d=s.d;s.lp=lp
        tr=[]
        if method in('ft','scratch'): tr=list(s.ps)
        if method=='bias': tr=[l[k] for l in s.layers for k in('ab','bb')]
        if method=='top': tr=[v for v in s.layers[-1].values()]
        if method=='lora':
            for l in s.layers:
                e={}
                for k in which:
                    A=T(rg.normal(0,1/np.sqrt(d),(r,d)));B=T(np.zeros((d,r)));e[k]=(A,B);tr+=[A,B]
                s.lora.append(e)
        if method=='adapter':
            for l in s.layers:
                e=[(T(rg.normal(0,1/np.sqrt(d),(d,r))),T(np.zeros((r,d))),T(np.zeros(d))) for _ in range(2)];s.ad.append(e);tr+=[p for x in e for p in x]
        if method=='prefix':
            s.pf=T(rg.normal(0,.1,(lp,d)));tr=[s.pf]
        s.tr=tr;return tr
    def W(s,i,l,k):
        if s.method=='lora' and k in s.lora[i]:
            A,B=s.lora[i][k];return l[k]+(B@A)*(4.0/s.r)     # α=4
        return l[k]
    def attn2(s,i,l,x):
        B,n,_=x.v.shape;h,dk=s.h,s.d//s.h
        sp=lambda t:t.rs(B,n,h,dk).t(0,2,1,3)
        Q,K,W=sp(x@s.W(i,l,'q')),sp(x@s.W(i,l,'k')),sp(x@s.W(i,l,'v'))
        sc=Q@K.t(0,1,3,2)*(1/np.sqrt(dk));a=sc.softmax();s.last=a.v
        return (a@W).t(0,2,1,3).rs(B,n,h*dk)@s.W(i,l,'o')
    def adp(s,i,j,y):
        if s.method!='adapter':return y
        U,D,b=s.ad[i][j];return y+(((y@U).relu()@D)+b)
    def forward(s,x):
        B,n=x.shape
        if s.pf is None: h=T(np.eye(s.V)[x])@s.E*np.sqrt(s.d)+T(np.eye(s.L)[:n])@s.P;off=0
        else:
            off=s.lp;m=T(np.eye(n+off)[:,:off])     # 前 lp 个位置放可学习向量（内容 PAD 不用）
            xx=np.c_[np.full((B,off),PAD),x];h=T(np.eye(s.V)[xx])@s.E*np.sqrt(s.d)+T(np.eye(s.L)[:n+off])@s.P+(m@s.pf)
        for i,l in enumerate(s.layers):
            h=(h+s.adp(i,0,s.attn2(i,l,h))).lnorm();y=(((h@l['a'])+l['ab']).relu()@l['b'])+l['bb'];h=(h+s.adp(i,1,y)).lnorm()
        return h@s.E.t(1,0),off
    def step(s,x,y,w,opt,lr):
        lg,off=s.forward(x)
        if off:
            B=x.shape[0];y=np.c_[np.zeros((B,off),int),y];w=np.c_[np.zeros((B,off)),w]
        val,p,sd=ce(lg,y,w);p.backward(sd)
        for q in s.tr:
            if q.g is None:q.g=np.zeros_like(q.v)
        keep=set(id(q) for q in s.tr)
        for q in s.ps:
            if id(q) not in keep:q.g=None
        opt.step(lr);return val
    def acc(s,task,B=300,seed=9):
        r=np.random.default_rng(seed);x,y,w=gen(r,B,task)
        if task==DOWN:x=downx(x)
        lg,off=s.forward(x);p=lg.v[:,off:].argmax(-1);return float(((p==y)*w).sum()/w.sum()),float(((p==y)|(w==0)).all(1).mean())
def base(seed,steps):
    m=M(V,L,d=32,h=4,ff=64,N=2,seed=seed);m.setup('ft');o=Adam(m.tr);r=np.random.default_rng(seed+50);t=time.time()
    for st in range(1,steps+1):
        x,y,w=gen(r,32,st%4);m.step(x,y,w,o,min(2e-3,2e-3*st/200)*(.3 if st>.8*steps else 1))
        if st%500==0:print('base',st,[round(m.acc(k)[0],3) for k in range(4)],m.acc(DOWN)[0],round(time.time()-t),flush=True)
    return [p.v.copy() for p in m.ps]
def clone(w,seed):
    m=M(V,L,d=32,h=4,ff=64,N=2,seed=seed)
    for p,v in zip(m.ps,w):p.v=v.copy()
    return m
def adapt(w,method,seed,steps,nd=None,r=4,which=('q','v'),lp=0,lr=2e-3):
    m=clone(w,seed) if method!='scratch' else M(V,L,d=32,h=4,ff=64,N=2,seed=seed+77);tr=m.setup(method,r,seed,which,lp)
    rg=np.random.default_rng(seed+500);o=Adam(tr) if tr else None;t0=time.time()
    if nd: X,Y,Wt=gen(rg,nd,DOWN);X=downx(X)
    if steps and tr:
        for st in range(1,steps+1):
            if nd:
                i=rg.integers(0,nd,32);x,y,wt=X[i],Y[i],Wt[i]
            else:
                x,y,wt=gen(rg,32,DOWN);x=downx(x)
            m.step(x,y,wt,o,lr*min(1,st/50)*(.3 if st>.8*steps else 1))
    a=m.acc(DOWN);return m,dict(acc=a[0],full=a[1],np=int(sum(p.v.size for p in tr)),sec=round(time.time()-t0,1))
def phi(A,B,i,j):
    Ua=np.linalg.svd(A,full_matrices=False)[2].T[:,:i];Ub=np.linalg.svd(B,full_matrices=False)[2].T[:,:j];return float((np.linalg.norm(Ua.T@Ub)**2)/min(i,j))
if __name__=='__main__':
    sd=int(sys.argv[1]);bs=int(sys.argv[2]) if len(sys.argv)>2 else 2500;ad=int(sys.argv[3]) if len(sys.argv)>3 else 800
    if bs<=0:
        m=M(V,L,d=32,h=4,ff=64,N=2,seed=1);m.setup('lora',2);x,y,w=gen(np.random.default_rng(0),2,4);x=downx(x)
        # 把 B 设非零再查 A、B 的梯度
        for e in m.lora:
            for k in e:e[k][1].v=np.random.default_rng(3).normal(0,.1,e[k][1].v.shape)
        lg,_=m.forward(x);v,p,g=ce(lg,y,w);p.backward(g);P=m.lora[0]['q'][0];a=P.g[1,2];E=1e-5
        def f():
            return ce(m.forward(x)[0],y,w)[0]
        P.v[1,2]+=E;v1=f();P.v[1,2]-=2*E;v2=f();P.v[1,2]+=E;print('grad',abs(a-(v1-v2)/2/E));sys.exit()
    w=base(sd,bs);out={'base':None}
    pm=clone(w,sd);out['base']=[pm.acc(k)[0] for k in range(4)]
    runs=[('不适配','none',{}),('从零训练','scratch',{}),('全量微调','ft',{'lr':5e-4}),('只训偏置','bias',{}),('只训最后一层','top',{}),('前缀 4 个','prefix',{'lp':4}),('前缀 16 个','prefix',{'lp':16}),('适配器 r=4','adapter',{'r':4}),('LoRA r=1','lora',{'r':1}),('LoRA r=2','lora',{'r':2}),('LoRA r=4','lora',{'r':4}),('LoRA r=8','lora',{'r':8}),('LoRA r=32','lora',{'r':32}),('LoRA 只q r=8','lora',{'r':8,'which':('q',)}),('LoRA 四矩阵 r=2','lora',{'r':2,'which':('q','k','v','o')})]
    keep={}
    for n,me,k in runs:
        steps=0 if me=='none' else ad;m,res=adapt(w,me,sd,steps,**k);out[n]=res;print(sd,n,res,flush=True);keep[n]=m
        json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
    # 数据很少：100 条
    out['low']={}
    for n,me,k in (('全量微调','ft',{'lr':5e-4}),('只训偏置','bias',{}),('前缀 16 个','prefix',{'lp':16}),('LoRA r=4','lora',{'r':4})):
        for nd in (30,100,1000):
            m,res=adapt(w,me,sd,ad,nd=nd,**k);out['low'][f'{n}/{nd}']=res;print(sd,n,nd,res,flush=True)
    # 全量微调学到的 ΔW 是低秩吗（奇异值）；LoRA 不同秩、不同种子的子空间；放大倍数
    ft=keep['全量微调'];b0=clone(w,sd);sv={}
    for i,(l,l0) in enumerate(zip(ft.layers,b0.layers)):
        for k in('q','v'):
            D=l[k].v-l0[k].v;sv[f'{i}{k}']=np.linalg.svd(D,compute_uv=False).round(4).tolist()
    out['ft_sv']=sv;an={}
    m8=keep['LoRA r=8'];m32=keep['LoRA r=32'];m2,_=adapt(w,'lora',sd+9,ad,r=8)
    for k in('q','v'):
        A8,A32,A82=m8.lora[0][k][0].v,m32.lora[0][k][0].v,m2.lora[0][k][0].v
        an[k]=dict(r8_32_top1=phi(A8,A32,1,1),r8_32_top8=phi(A8,A32,8,8),seed_top1=phi(A8,A82,1,1),seed_top4=phi(A8,A82,4,4),rand_top4=phi(np.random.default_rng(0).normal(size=(8,32)),np.random.default_rng(1).normal(size=(8,32)),4,4))
    for rn,mm in(('r1',keep['LoRA r=1']),('r4',keep['LoRA r=4']),('r8',m8),('r32',m32)):
        for k in('q','v'):
            A,B=mm.lora[0][k];D=(B.v@A.v)*(4.0/mm.r);W0=b0.layers[0][k].v;U,S,Vt=np.linalg.svd(D);q=mm.r
            pj=np.linalg.norm(U[:,:q].T@W0@Vt[:q].T);tu,_,tv=np.linalg.svd(W0);top=np.linalg.norm(tu[:,:q].T@W0@tv[:q].T)
            an[f'{rn}{k}']=dict(dw=float(np.linalg.norm(D)),proj=float(pj),top=float(top),w=float(np.linalg.norm(W0)),amp=float(np.linalg.norm(D)/pj))
    out['an']=an
    json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
    if sd==0:
        s={f'base{i}':p for i,p in enumerate(w)}
        for nm in('LoRA r=1','LoRA r=4','LoRA r=8'):
            mm=keep[nm];tag=nm.split('=')[1]
            for i,e in enumerate(mm.lora):
                for k in e:
                    s[f'r{tag}_{i}{k}']=mm.layers[i][k].v+(e[k][1].v@e[k][0].v)*(4.0/mm.r)
        np.savez('lora.npz',**s)
