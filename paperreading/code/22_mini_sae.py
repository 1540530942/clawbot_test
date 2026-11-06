#!/usr/bin/env python3
# 稀疏自编码器：有标准答案的叠加玩具，和字符级 Transformer 上的特征分解
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

"""第 22 期玩具实验：A 有标准答案的“叠加”：把很多稀疏真特征压进少量神经元，看稀疏自编码器能否找回。
B 真模型：字符级单层 Transformer 的 MLP 激活，训练稀疏自编码器，看特征比神经元是否更单一。"""
import sys,json,glob,re,time,numpy as np
class SAE:
    def __init__(s,n,m,seed):
        r=np.random.default_rng(seed+12345);s.Wd=r.normal(size=(n,m));s.Wd/=np.linalg.norm(s.Wd,axis=0,keepdims=True);s.We=s.Wd.T.copy();s.be=np.zeros(m);s.bd=np.zeros(n);s.st={}
    def enc(s,X):return np.maximum((X-s.bd)@s.We.T+s.be,0)
    def dec(s,F):return F@s.Wd.T+s.bd
    def step(s,X,lam,lr):
        F=s.enc(X);R=s.dec(F);E=R-X;B=len(X)
        act=(F>0).astype(float)
        gWd=2*E.T@F/B/X.shape[1]
        G=2*E@s.Wd/X.shape[1]+lam        # dLoss/dF（只对激活的单元）
        gWe=(((G*act).T)@(X-s.bd))/B
        gbe=(G*act).sum(0)/B
        gbd=2*E.mean(0)/X.shape[1]-((G*act)@s.We).sum(0)/B
        for k,g in(('Wd',gWd),('We',gWe),('be',gbe),('bd',gbd)):
            m,v,t=s.st.get(k,(0,0,0));t+=1;m=.9*m+.1*g;v=.999*v+.001*g*g;s.st[k]=(m,v,t);setattr(s,k,getattr(s,k)-lr*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8))
        s.Wd/=np.linalg.norm(s.Wd,axis=0,keepdims=True)+1e-9
        return float((E*E).mean()),float(act.sum(1).mean())
def toy(seed,out):
    n,K=16,64;r=np.random.default_rng(seed);D=r.normal(size=(n,K));D/=np.linalg.norm(D,axis=0,keepdims=True)
    def sample(B,p=.03):
        A=(r.random((B,K))<p)*r.random((B,K));return A@D.T,A
    # 叠加下，神经元有多“杂”：每个神经元权重最大的前 5 个特征占能量比例
    res=[];X0,A0=sample(4000)
    for mult in(1,2,4,8):
        for lam in(.001,.003,.01,.03):
            s=SAE(n,n*mult,seed)
            for st in range(1,4001):
                X,_=sample(256);mse,l0=s.step(X,lam,1e-3*(1 if st<3000 else .2))
            C=np.abs(D.T@s.Wd);rec=C.max(1);F=s.enc(X0);mse=float(((s.dec(F)-X0)**2).mean());base=float(X0.var(0).sum()/n)
            res.append(dict(dict=n*mult,lam=lam,mmcs=float(rec.mean()),rec9=float((rec>.9).mean()),mse=mse/base,l0=float((F>0).sum(1).mean()),dead=float((F.max(0)==0).mean())));print('toy',seed,res[-1],flush=True)
    out['toy']=res;out['toy_true_l0']=float((A0>0).sum(1).mean())
    # 神经元层面的相关：一个神经元和最相关特征的相关性
    out['toy_neuron_corr']=float(np.mean([np.max(np.abs(np.corrcoef(X0[:,i],A0.T)[0,1:])) for i in range(n)]))
def corpus():
    t=''
    for f in ['01','02','05','06','07']:
        s=open(f'../../papers/{f}/paper.txt',encoding='utf-8').read();s=s[:s.find('\nReferences')] if '\nReferences' in s else s
        s='\n'.join(l for l in s.split('\n') if len(l)>80 and not l.strip().startswith('|') and '$' not in l);t+=s+'\n'
    return re.sub(r'[^\x20-\x7e\n]','',t)
CTX=32
CLASS={'数字':lambda c:c.isdigit(),'大写':lambda c:c.isupper(),'空格':lambda c:c==' ','标点':lambda c:c in '.,;:','括号':lambda c:c in '()[]','元音':lambda c:c in 'aeiou','换行':lambda c:c=='\n'}
def mlp_in(m,x):
    n=x.shape[1];h=T(np.eye(m.V)[x])@m.E*np.sqrt(m.d)+T(np.eye(m.L)[:n])@m.P;l=m.layers[0];return (h+m.attn(l,h)).lnorm().v,l
def after(h,F,m,l):
    y=F@l['b'].v+l['bb'].v+h;mm=y.mean(-1,keepdims=True);y=(y-mm)/np.sqrt(((y-mm)**2).mean(-1,keepdims=True)+1e-6);return y@m.E.v.T
def real(seed,out,S):
    tx=corpus();CH=sorted(set(tx));ID={c:i for i,c in enumerate(CH)};A=np.array([ID[c] for c in tx]);cut=int(len(A)*.9);TR,TE=A[:cut],A[cut:];V=len(CH)
    m=Enc(V,CTX,d=32,h=4,ff=256,N=1,seed=seed,causal=True);o=Adam(m.ps);r=np.random.default_rng(seed+1)
    def get(s,B):
        i=r.integers(0,len(s)-CTX-1,B);return np.stack([s[j:j+CTX] for j in i]),np.stack([s[j+1:j+CTX+1] for j in i])
    for st in range(1,S+1):
        x,y=get(TR,32);v,p,g=ce(m.lm(x),y);p.backward(g);o.step(2e-3*min(1,st/100)*(.1+.45*(1+np.cos(np.pi*st/S))))
    print('real lm',round(float(v),3),flush=True)
    def acts(s,N):
        Xs=[];H=[];Ls=[];Ys=[]
        for _ in range(N):
            x,y=get(s,64);h,l=mlp_in(m,x);F=np.maximum(h@l['a'].v+l['ab'].v,0);Xs.append(F.reshape(-1,F.shape[-1]).astype(np.float32));H.append(h);Ls.append(x);Ys.append(y)
        return Xs,H,Ls,Ys
    Xs,_,_,_=acts(TR,100);D=np.concatenate(Xs);del Xs;sc=np.sqrt((D**2).sum(1).mean());D=D/sc*np.sqrt(256);print(D.shape,flush=True)
    s=SAE(256,256*4,seed);r2=np.random.default_rng(seed+2);lam=.1 if False else .02
    for st in range(1,6001):
        X=D[r2.integers(0,len(D),512)];mse,l0=s.step(X,lam,3e-4*(1 if st<5000 else .2))
    # 损失恢复比例
    Xt,Ht,Lt,Yt=acts(TE,15);base=[];rec=[];zero=[]
    def ll(lg,y):
        z=lg-lg.max(-1,keepdims=True);p=np.exp(z);p/=p.sum(-1,keepdims=True);return float(-np.log(p[np.arange(y.shape[0])[:,None],np.arange(y.shape[1])[None,:],y]+1e-12).mean())
    l=m.layers[0]
    for F,h,y in zip(Xt,Ht,Yt):
        F=F.reshape(h.shape[0],h.shape[1],-1);base.append(ll(after(h,F,m,l),y));G=s.dec(s.enc(F/sc*np.sqrt(256)))*sc/np.sqrt(256);rec.append(ll(after(h,G,m,l),y));zero.append(ll(after(h,0*F,m,l),y))
    b,rc,z=map(np.mean,(base,rec,zero));out['loss']=dict(base=float(b),recon=float(rc),zero=float(z),recovered=float((z-rc)/(z-b)))
    # 特征与神经元对“字符类别”的相关
    Xall=np.concatenate(Xt)/sc*np.sqrt(256);FF=s.enc(Xall).astype(np.float32);chars=np.concatenate([x.reshape(-1) for x in Lt]);ch=[CH[i] for i in chars]
    Lab=np.stack([[fn(c) for c in ch] for fn in CLASS.values()]).astype(float);names=list(CLASS)
    def best(Y):
        Y=Y[:,Y.std(0)>1e-9];Yz=(Y-Y.mean(0))/Y.std(0);Lz=(Lab-Lab.mean(1,keepdims=True))/(Lab.std(1,keepdims=True)+1e-9);C=np.abs(Lz@Yz/len(ch));return C
    Cf,Cn=best(FF),best(Xall);topF=np.sort(Cf,axis=1)[:,-1];topN=np.sort(Cn,axis=1)[:,-1]
    out['class']={n:dict(feature=float(a),neuron=float(b)) for n,a,b in zip(names,topF,topN)}
    act=(FF>0).mean(0);out['sae']=dict(dict=256*4,l0=float((FF>0).sum(1).mean()),dead=float((act==0).mean()),neuron_l0=float((Xall>0).sum(1).mean()))
    # 上下文样例：每个特征激活最高的片段
    txt=[''.join(CH[i] for i in x) for b in Lt for x in b];N=len(txt);cl=CTX
    def tops(col,k=6):
        i=np.argsort(col)[::-1][:k];o2=[]
        for j in i:
            row,pos=j//cl,j%cl;s0=txt[row];o2.append([float(round(col[j],2)),s0[max(0,pos-14):pos],s0[pos],s0[pos+1:pos+4]])
        return o2
    alive=np.where(act>.0005)[0];rs=np.random.default_rng(seed+3);pick=rs.choice(alive,min(24,len(alive)),replace=False);out['features']=[dict(id=int(i),density=float(act[i]),top=tops(FF[:,i])) for i in pick]
    nr=rs.choice(256,8,replace=False);out['neurons']=[dict(id=int(i),density=float((Xall[:,i]>0).mean()),top=tops(Xall[:,i])) for i in nr]
    # 对每个字符类别，挑最相关的特征和神经元的样例
    best_f={};Ccols=np.where(FF.std(0)>1e-9)[0];Cn2=np.where(Xall.std(0)>1e-9)[0]
    for k,n in enumerate(names):
        i=Ccols[int(np.argmax(Cf[k]))];j=Cn2[int(np.argmax(Cn[k]))];best_f[n]=dict(feature=dict(id=int(i),top=tops(FF[:,i])),neuron=dict(id=int(j),top=tops(Xall[:,j])))
    out['best']=best_f
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 2500;out={};t=time.time()
    if S<=0:
        s=SAE(8,16,0);X=np.random.default_rng(0).random((20,8));s.bd=X.mean(0);F=s.enc(X);print(s.step(X,.01,1e-3));sys.exit()
    toy(sd,out);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
    if S==1:print('只跑玩具部分');sys.exit()
    real(sd,out,S);out['sec']=round(time.time()-t);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False);print(out['loss'],out['class'],out['sae'],out['sec'])
