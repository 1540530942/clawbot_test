#!/usr/bin/env python3
# 第 50 期 GPT-4 玩具实验：可预测的扩展、校准和子串污染检测
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

"""第 50 期玩具实验：GPT-4 报告里可以自己动手的三件事。一是可预测的扩展：只用小模型拟合 L(C)=aC^b+c，
先把预测写进文件，再训练大模型核对。二是校准：预训练模型的信心和准确率对得上吗，用一种“只奖励答对一个选项”的后训练会不会变差。
三是用论文的子串方法查污染，看改一个词能否逃过。语料是这些论文的英文全文，可换成任意英文文本。这是玩具。"""
import sys,json,time,glob,re,numpy as np
CTX=12;EMB=8
def corpus():
    t=' '.join(open(f,encoding='utf-8',errors='ignore').read() for f in sorted(glob.glob('../../papers/*/compact.txt')));t=re.sub(r'\s+',' ',t.lower());t=''.join(c for c in t if 32<=ord(c)<127);t=re.sub(r'[^a-z0-9 .,;:()\-]','',t);return t
class LM:
    def __init__(s,V,H,seed):
        r=np.random.default_rng(seed);s.E=r.normal(0,.1,(V,EMB));s.W=[r.normal(0,1/np.sqrt(CTX*EMB),(CTX*EMB,H)),r.normal(0,1/np.sqrt(H),(H,H)),r.normal(0,1/np.sqrt(H),(H,V))];s.b=[np.zeros(H),np.zeros(H),np.zeros(V)];s.V=V
    def fw(s,x):
        h=s.E[x].reshape(len(x),-1);hs=[h]
        for i in range(3):
            y=h@s.W[i]+s.b[i]
            if i<2:y=np.maximum(y,0)
            hs.append(y);h=y
        return h,hs
def train(m,D,tr,steps,seed,lr0=2e-3,lab=None):
    """lab：给定就用这个标签（用于后训练）"""
    r=np.random.default_rng(seed);P=[m.E]+m.W+m.b;M=[np.zeros_like(p) for p in P];U=[np.zeros_like(p) for p in P];B=128
    for st in range(1,steps+1):
        ii=r.integers(0,len(tr),B);j=tr[ii];x=D[j[:,None]-np.arange(CTX,0,-1)];y=D[j] if lab is None else lab[ii];lg,hs=m.fw(x);e=lg-lg.max(1,keepdims=True);p=np.exp(e);p/=p.sum(1,keepdims=True);p[np.arange(B),y]-=1;u=p;G=[None]*7;G[3]=hs[2].T@u/B;G[6]=u.mean(0);u=u@m.W[2].T
        for i in (1,0):
            u=u*(hs[i+1]>0);G[1+i]=hs[i].T@u/B;G[4+i]=u.mean(0);u=u@m.W[i].T
        gx=u.reshape(B,CTX,EMB);ge=np.zeros_like(m.E);np.add.at(ge,x.ravel(),gx.reshape(-1,EMB)/B);G[0]=ge;lr=lr0 if st<.8*steps else lr0/5
        for k,(q,g) in enumerate(zip(P,G)):
            M[k]=.9*M[k]+.1*g;U[k]=.999*U[k]+.001*g*g;q-=lr*(M[k]/(1-.9**st))/(np.sqrt(U[k]/(1-.999**st))+1e-8)
def nll(m,D,va):
    t=0.
    for i in range(0,len(va),8192):
        j=va[i:i+8192];lg=m.fw(D[j[:,None]-np.arange(CTX,0,-1)])[0];e=lg-lg.max(1,keepdims=True);t+=-(e[np.arange(len(j)),D[j]]-np.log(np.exp(e).sum(1))).sum()
    return float(t/len(va))
def fit(C,L):
    """L=a C^b + c：对 c 做网格，其余在 log 上线性回归"""
    C,L=np.array(C,float),np.array(L,float);best=None
    for c in np.linspace(0,L.min()-1e-3,200):
        X=np.c_[np.ones(len(C)),np.log(C)];y=np.log(L-c);w,*_=np.linalg.lstsq(X,y,rcond=None);e=((np.exp(X@w)+c-L)**2).sum()
        if best is None or e<best[0]:best=(e,float(np.exp(w[0])),float(w[1]),float(c))
    return best[1:]
def calib(m,D,va,bins=10):
    """多选题校准：每条取 4 个选项（正确的 + 3 个其它字符），信心分箱"""
    r=np.random.default_rng(3);j=va[:20000];lg=m.fw(D[j[:,None]-np.arange(CTX,0,-1)])[0];n=len(j);opt=np.zeros((n,4),int)
    for i in range(n):
        o=[int(D[j[i]])]+[int(v) for v in r.choice([c for c in range(m.V) if c!=D[j[i]]],3,replace=False)];opt[i]=o
    q=np.take_along_axis(lg,opt,1);p=np.exp(q-q.max(1,keepdims=True));p/=p.sum(1,keepdims=True);conf=p.flatten();cor=np.zeros_like(p);cor[:,0]=1;cor=cor.flatten();b=np.clip((conf*bins).astype(int),0,bins-1);row=[];ece=0
    for k in range(bins):
        s=b==k
        if s.sum():row.append(dict(c=float(conf[s].mean()),a=float(cor[s].mean()),n=int(s.sum())));ece+=s.mean()*abs(conf[s].mean()-cor[s].mean())
    return dict(rows=row,ece=float(ece),acc=float((p.argmax(1)==0).mean()))
def contamination(T_,seed):
    """附录 C 的方法：去掉空格和符号，随机取 3 段 50 字符，看是否是训练集子串"""
    r=np.random.default_rng(seed);clean=re.sub(r'[^a-z0-9]','',T_);tr=clean[:len(clean)//2];te=clean[len(clean)//2:];n=200;res={}
    def hit(ex):
        for _ in range(3):
            if len(ex)<=50:
                if ex in tr:return True
            else:
                i=int(r.integers(0,len(ex)-50));
                if ex[i:i+50] in tr:return True
        return False
    S=[tr[i:i+300] for i in r.integers(0,len(tr)-300,n)];res['copied']=float(np.mean([hit(s) for s in S]))
    def edit(s,k):
        s=list(s)
        for i in r.integers(0,len(s),k):s[i]='z' if s[i]!='z' else 'q'
        return ''.join(s)
    res['edit']={str(k):float(np.mean([hit(edit(s,k)) for s in S])) for k in (1,5,10,20,50)}
    Sn=[te[i:i+300] for i in r.integers(0,len(te)-300,n)];res['fresh']=float(np.mean([hit(s) for s in Sn]));return res
if __name__=='__main__':
    sd=int(sys.argv[1]);LargeS=int(sys.argv[2]) if len(sys.argv)>2 else 16000;t0=time.time();T_=corpus();ch=sorted(set(T_));cid={c:i for i,c in enumerate(ch)};D=np.array([cid[c] for c in T_],np.int32);V=len(ch);r=np.random.default_rng(sd);idx=np.arange(CTX+1,len(D));r.shuffle(idx);tr,va=idx[:-60000],idx[-50000:];res={'chars':len(D),'V':V}
    # 训练量和宽度一起增长。算力 ∝ H²·步数
    G=[(16,LargeS//64),(16,LargeS//32),(24,LargeS//32),(32,LargeS//16),(32,LargeS//8),(48,LargeS//8),(64,LargeS//8),(64,LargeS//4),(96,LargeS//4)];C=[];L=[]
    Hbig=256;Cbig=Hbig**2*LargeS
    for H,s in G:
        m=LM(V,H,sd+H);train(m,D,tr,s,sd+s);C.append(H**2*s/Cbig);L.append(nll(m,D,va));print(H,s,round(C[-1],6),round(L[-1],3),round(time.time()-t0),flush=True)
    res['small']=dict(H=[g[0] for g in G],steps=[g[1] for g in G],C=C,L=L);a,b,c=fit(C,L);pred=a*1.+c;res['fit']=dict(a=a,b=b,c=c,pred=pred);res['fit_noc']=None;res['ratio']=1/max(C)
    # 先把预测写进文件
    json.dump(dict(pred=pred,a=a,b=b,c=c,when=time.time(),maxC=max(C)),open(f'pred{sd}.json','w'));print('预测',pred,round(time.time()-t0),flush=True)
    Big=LM(V,Hbig,sd+999);train(Big,D,tr,LargeS,sd+1000);res['big']=nll(Big,D,va);print('实际',res['big'],round(time.time()-t0),flush=True)
    # 校准
    res['cal0']=calib(Big,D,va);print('校准前',res['cal0']['ece'],res['cal0']['acc'],flush=True)
    # 后训练：把模型最想说的那个字符当标签再训练（只奖励一个答案），模仿“强化”带来的过度自信
    j=tr[:60000];lg=Big.fw(D[j[:,None]-np.arange(CTX,0,-1)])[0];lab=lg.argmax(1);trsub=j;train(Big,D,trsub,LargeS//8,sd+1001,lr0=3e-4,lab=lab);res['cal1']=calib(Big,D,va);res['big_post']=nll(Big,D,va);print('校准后',res['cal1']['ece'],res['cal1']['acc'],res['big_post'],flush=True)
    res['cont']=contamination(T_,sd+2);print(res['cont'],flush=True)
    res['sec']=time.time()-t0;json.dump(res,open(f'res{sd}.json','w'));print('done',round(time.time()-t0),flush=True)
