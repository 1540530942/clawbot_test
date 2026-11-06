#!/usr/bin/env python3
# 第 44 期 GPTQ 玩具实验：RTN、GPTQ 和贪心 OBQ，块大小不变性，校准量和阻尼
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

"""第 44 期玩具实验：GPTQ。训练一个字符级语言模型（MLP，语料是这些论文的英文全文），再把权重压到 2/3/4 比特。
比较直接取整（RTN）、GPTQ（算法 1）、贪心顺序的 OBQ。校准数据、阻尼、是否顺序传递量化后的输入、分组。
Y=X W：这里每列一个输出，H=2XᵀX 只和输入有关，所以所有输出共用一份 H。语料可以换成任意英文文本。这是玩具。"""
import sys,json,time,glob,re,numpy as np
CTX=16;EMB=12;H=128
def corpus():
    t=' '.join(open(f,encoding='utf-8',errors='ignore').read() for f in sorted(glob.glob('../../papers/*/compact.txt')));t=re.sub(r'\s+',' ',t.lower());t=''.join(c for c in t if 32<=ord(c)<127);t=re.sub(r'[^a-z0-9 .,;:()\-]','',t);return t
class LM:
    def __init__(s,V,seed):
        r=np.random.default_rng(seed);s.E=r.normal(0,.1,(V,EMB));s.W=[r.normal(0,1/np.sqrt(CTX*EMB),(CTX*EMB,H)),r.normal(0,1/np.sqrt(H),(H,H)),r.normal(0,1/np.sqrt(H),(H,V))];s.b=[np.zeros(H),np.zeros(H),np.zeros(V)];s.V=V
    def fw(s,x,W=None):
        W=W or s.W;h=s.E[x].reshape(len(x),-1);hs=[h]
        for i in range(3):
            y=h@W[i]+s.b[i]
            if i<2:y=np.maximum(y,0)
            hs.append(y);h=y
        return h,hs
def nll(lg,y):
    e=lg-lg.max(1,keepdims=True);return float(-(e[np.arange(len(y)),y]-np.log(np.exp(e).sum(1))).mean())
def ppl(m,D,idx,W=None):
    t=0.
    for i in range(0,len(idx),4096):
        j=idx[i:i+4096];x=D[j[:,None]-np.arange(CTX,0,-1)];t+=nll(m.fw(x,W)[0],D[j])*len(j)
    return float(np.exp(t/len(idx)))
def train(m,D,tr,steps,seed):
    r=np.random.default_rng(seed);P=[m.E]+m.W+m.b;M=[np.zeros_like(p) for p in P];U=[np.zeros_like(p) for p in P]
    for st in range(1,steps+1):
        j=tr[r.integers(0,len(tr),128)];x=D[j[:,None]-np.arange(CTX,0,-1)];y=D[j];lg,hs=m.fw(x);e=lg-lg.max(1,keepdims=True);p=np.exp(e);p/=p.sum(1,keepdims=True);p[np.arange(128),y]-=1;u=p/1.;G=[None]*7
        G[3]=hs[2].T@u/128;G[6]=u.mean(0);u=u@m.W[2].T
        for i in (1,0):
            u=u*(hs[i+1]>0);G[1+i]=hs[i].T@u/128;G[4+i]=u.mean(0);u=u@m.W[i].T
        gx=u.reshape(128,CTX,EMB);ge=np.zeros_like(m.E);np.add.at(ge,x.ravel(),gx.reshape(-1,EMB)/128);G[0]=ge;lr=2e-3 if st<.8*steps else 4e-4
        for k,(q,g) in enumerate(zip(P,G)):
            M[k]=.9*M[k]+.1*g;U[k]=.999*U[k]+.001*g*g;q-=lr*(M[k]/(1-.9**st))/(np.sqrt(U[k]/(1-.999**st))+1e-8)
class Grid:
    """每个输出（这里是每列）一套非对称 min-max 网格。g>0 时每 g 个权重一个网格"""
    def __init__(s,w,bits,g=0):
        s.L=2**bits-1;s.g=g
        if g:
            n=w.shape[0];s.lo=np.stack([np.tile(w[i:i+g].min(0),(min(g,n-i),1)) for i in range(0,n,g)]) if False else np.concatenate([np.tile(w[i:i+g].min(0),(len(w[i:i+g]),1)) for i in range(0,n,g)]);s.hi=np.concatenate([np.tile(w[i:i+g].max(0),(len(w[i:i+g]),1)) for i in range(0,n,g)])
        else:
            s.lo=np.tile(w.min(0),(w.shape[0],1));s.hi=np.tile(w.max(0),(w.shape[0],1))
    def q(s,w,j):
        lo,hi=s.lo[j],s.hi[j];st=(hi-lo)/s.L+1e-12;return lo+np.clip(np.round((w-lo)/st),0,s.L)*st
def rtn(W,bits,g=0):
    G=Grid(W,bits,g);o=np.zeros_like(W)
    for j in range(W.shape[0]):o[j]=G.q(W[j],j)
    return o
def gptq(W,X,bits,B=32,damp=.01,g=0,order=None):
    """算法 1。W:(in,out)，X:(m,in)。按输入维度依次量化，所有输出同一顺序"""
    W=W.copy();n=W.shape[0];Gd=Grid(W,bits,g);Hs=2*X.T@X;Hs=Hs+damp*np.mean(np.diag(Hs))*np.eye(n)+1e-8*np.eye(n)
    p=np.arange(n) if order is None else order;Hp=Hs[np.ix_(p,p)];Hi=np.linalg.cholesky(np.linalg.inv(Hp)).T   # 上三角 U，U 的第 j 行从对角线起就是消元到 j 时 H⁻¹ 的第 j 行除以根号
    Wp=W[p];Q=np.zeros_like(Wp);Gdp=Grid.__new__(Grid);Gdp.L=Gd.L;Gdp.lo=Gd.lo[p];Gdp.hi=Gd.hi[p]
    for i in range(0,n,B):
        e=min(n,i+B);E=np.zeros((e-i,W.shape[1]))
        for j in range(i,e):
            Q[j]=Gdp.q(Wp[j],j);E[j-i]=(Wp[j]-Q[j])/Hi[j,j];Wp[j:e]-=Hi[j,j:e][:,None]*E[j-i][None,:]
        if e<n:Wp[e:]-=Hi[i:e,e:].T@E
    o=np.zeros_like(W);o[p]=Q;return o
def obq(W,X,bits,damp=.01):
    """贪心顺序：每列单独，每次选当前代价最小的权重（式 2、3）。慢，只用于小层"""
    n,c=W.shape;Hs=2*X.T@X;Hs=Hs+damp*np.mean(np.diag(Hs))*np.eye(n)+1e-8*np.eye(n);H0=np.linalg.inv(Hs);O=np.zeros_like(W)
    Gd=Grid(W,bits)
    for k in range(c):
        Hi=H0.copy();w=W[:,k].copy();F=np.ones(n,bool);Lo,Hi2=Gd.lo[:,k],Gd.hi[:,k];st=(Hi2-Lo)/Gd.L+1e-12
        for _ in range(n):
            q=Lo+np.clip(np.round((w-Lo)/st),0,Gd.L)*st;d=np.diag(Hi);c_=np.where(F,(q-w)**2/np.maximum(d,1e-15),np.inf);j=int(c_.argmin());de=(w[j]-q[j])/Hi[j,j]*Hi[:,j];w=w-de;w[j]=q[j];O[j,k]=q[j];Hi=Hi-np.outer(Hi[:,j],Hi[j,:])/Hi[j,j];F[j]=False;Hi[j,:]=0;Hi[:,j]=0
    return O
def quant_model(m,D,cal,bits,kind='gptq',seq=True,**k):
    """逐层量化。seq=True 时用前面已量化层的实际输出作为校准输入"""
    x=D[cal[:,None]-np.arange(CTX,0,-1)];W=[];t0=time.time()
    h=m.E[x].reshape(len(x),-1);h0=h
    for i in range(3):
        if kind=='rtn':q=rtn(m.W[i],bits,k.get('g',0))
        elif kind=='obq':q=obq(m.W[i],h,bits)
        else:q=gptq(m.W[i],h,bits,**k)
        W.append(q);src=q if seq else m.W[i];h=h@src+m.b[i]
        if i<2:h=np.maximum(h,0)
    return W,time.time()-t0
def layer_err(m,x,W):
    _,hs=m.fw(x);e=[]
    for i in range(3):
        e.append(float(((hs[i]@(m.W[i]-W[i]))**2).sum()/(hs[i]@m.W[i]).var()/len(x)))
    return e
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 12000;t0=time.time();r=np.random.default_rng(sd);T_=corpus();ch=sorted(set(T_));cid={c:i for i,c in enumerate(ch)};D=np.array([cid[c] for c in T_],np.int16);n=len(D);res={'chars':n,'V':len(ch),'vocab':''.join(ch)}
    idx=np.arange(CTX,n);r.shuffle(idx);tr,va,ca=idx[:n-60000],idx[-50000:],idx[-60000:-50000]
    m=LM(len(ch),sd);train(m,D,tr,S,sd+1);base=ppl(m,D,va);res['base']=base;print('base',base,round(time.time()-t0),flush=True)
    cal=ca[:2048];W8=[];res['q']={}
    for bits in (8,4,3,2):
        row={}
        for kn,kw in (('rtn',{}),('gptq',{})):
            W,sec=quant_model(m,D,cal,bits,kn,**kw);row[kn]=dict(ppl=ppl(m,D,va,W),sec=sec,err=layer_err(m,D[cal[:,None]-np.arange(CTX,0,-1)],W))
            if sd==0 and bits==3:np.savez(f'm_{kn}3.npz',E=m.E,**{f'W{i}':w for i,w in enumerate(W)},**{f'b{i}':b for i,b in enumerate(m.b)})
        res['q'][str(bits)]=row;print(bits,{k:round(v['ppl'],3) for k,v in row.items()},round(time.time()-t0),flush=True)
    if sd==0:np.savez('m_full.npz',E=m.E,**{f'W{i}':w for i,w in enumerate(m.W)},**{f'b{i}':b for i,b in enumerate(m.b)})
    # 贪心 OBQ（慢）
    res['obq']={}
    for bits in (4,3,2):
        W,sec=quant_model(m,D,cal[:512],bits,'obq');res['obq'][str(bits)]=dict(ppl=ppl(m,D,va,W),sec=sec);Wg,sg=quant_model(m,D,cal[:512],bits,'gptq');res['obq'][str(bits)]['gptq']=ppl(m,D,va,Wg);res['obq'][str(bits)]['gsec']=sg;print('obq',bits,res['obq'][str(bits)],round(time.time()-t0),flush=True)
    # 校准量
    res['cal']={str(c):ppl(m,D,va,quant_model(m,D,ca[:c],3)[0]) for c in (32,128,512,2048,8192)}
    res['damp']={str(d):ppl(m,D,va,quant_model(m,D,cal,3,damp=d)[0]) for d in (1e-6,1e-3,.01,.1,1.)}
    res['seq']=dict(on=ppl(m,D,va,quant_model(m,D,cal,3,seq=True)[0]),off=ppl(m,D,va,quant_model(m,D,cal,3,seq=False)[0]))
    res['block']={str(B):ppl(m,D,va,quant_model(m,D,cal,3,B=B)[0]) for B in (1,8,32,128)}
    pp=np.random.default_rng(1).permutation(CTX*EMB)
    res['group']={f'{b}/g{g}':dict(rtn=ppl(m,D,va,quant_model(m,D,cal,b,'rtn',g=g)[0]),gptq=ppl(m,D,va,quant_model(m,D,cal,b,g=g)[0])) for b,g in ((3,0),(3,32),(3,8),(2,32),(2,8))};print(res['cal'],res['damp'],res['seq'],res['block'],res['group'],flush=True)
    res['sec']=time.time()-t0;json.dump(res,open(f'res{sd}.json','w'));print('done',round(time.time()-t0),flush=True)
