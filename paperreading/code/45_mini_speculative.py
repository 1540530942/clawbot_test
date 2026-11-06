#!/usr/bin/env python3
# 第 45 期推测解码玩具实验：分布不变、接受率公式和宽容度
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

"""第 45 期玩具实验：推测解码。大小两个字符语言模型（语料是这些论文的英文全文，可换成任意英文文本），再加一元、二元表格模型。
测接受率 α，和公式对比每次调用大模型产生的字符数，验证输出分布真的不变，再试宽容度。真实 Transformer 的数字差很多，这是玩具。"""
import sys,json,time,glob,re,numpy as np
def corpus():
    t=' '.join(open(f,encoding='utf-8',errors='ignore').read() for f in sorted(glob.glob('../../papers/*/compact.txt')));t=re.sub(r'\s+',' ',t.lower());t=''.join(c for c in t if 32<=ord(c)<127);t=re.sub(r'[^a-z0-9 .,;:()\-]','',t);return t
class LM:
    def __init__(s,V,ctx,emb,H,seed):
        r=np.random.default_rng(seed);s.ctx,s.emb,s.V=ctx,emb,V;s.E=r.normal(0,.1,(V,emb));s.W=[r.normal(0,1/np.sqrt(ctx*emb),(ctx*emb,H)),r.normal(0,1/np.sqrt(H),(H,H)),r.normal(0,1/np.sqrt(H),(H,V))];s.b=[np.zeros(H),np.zeros(H),np.zeros(V)];s.flops=sum(w.size for w in s.W)
    def fw(s,x):
        h=s.E[x].reshape(len(x),-1);hs=[h]
        for i in range(3):
            y=h@s.W[i]+s.b[i]
            if i<2:y=np.maximum(y,0)
            hs.append(y);h=y
        return h,hs
    def logits(s,hist):
        """hist: (B,T) 最后 ctx 个字符"""
        return s.fw(hist[:,-s.ctx:])[0]
def train(m,D,tr,steps,seed):
    r=np.random.default_rng(seed);P=[m.E]+m.W+m.b;M=[np.zeros_like(p) for p in P];U=[np.zeros_like(p) for p in P];B=128
    for st in range(1,steps+1):
        j=tr[r.integers(0,len(tr),B)];x=D[j[:,None]-np.arange(m.ctx,0,-1)];y=D[j];lg,hs=m.fw(x);e=lg-lg.max(1,keepdims=True);p=np.exp(e);p/=p.sum(1,keepdims=True);p[np.arange(B),y]-=1;u=p;G=[None]*7;G[3]=hs[2].T@u/B;G[6]=u.mean(0);u=u@m.W[2].T
        for i in (1,0):
            u=u*(hs[i+1]>0);G[1+i]=hs[i].T@u/B;G[4+i]=u.mean(0);u=u@m.W[i].T
        gx=u.reshape(B,m.ctx,m.emb);ge=np.zeros_like(m.E);np.add.at(ge,x.ravel(),gx.reshape(-1,m.emb)/B);G[0]=ge;lr=2e-3 if st<.8*steps else 4e-4
        for k,(q,g) in enumerate(zip(P,G)):
            M[k]=.9*M[k]+.1*g;U[k]=.999*U[k]+.001*g*g;q-=lr*(M[k]/(1-.9**st))/(np.sqrt(U[k]/(1-.999**st))+1e-8)
class Gram:
    """一元或二元表格模型"""
    def __init__(s,D,V,n):
        s.n=n;s.V=V;s.t=np.ones((V,V) if n==2 else (1,V))
        if n==2:np.add.at(s.t,(D[:-1],D[1:]),1)
        else:np.add.at(s.t,(np.zeros(len(D),int),D),1)
        s.l=np.log(s.t/s.t.sum(1,keepdims=True));s.flops=1
    def logits(s,hist):
        return s.l[hist[:,-1]] if s.n==2 else np.tile(s.l[0],(len(hist),1))
def prob(l,T):
    """标准化采样：温度；T=0 是 argmax"""
    if T==0:
        o=np.zeros_like(l);o[np.arange(len(l)),l.argmax(1)]=1;return o
    e=(l-l.max(1,keepdims=True))/T;p=np.exp(e);return p/p.sum(1,keepdims=True)
def spec(Mp,Mq,hist,gamma,T,r,lenience=1.):
    """算法 1。hist 一维数组。返回新字符列表（1 到 γ+1 个）、接受个数"""
    h=list(hist);G=[];Q=[]
    for i in range(gamma):
        q=prob(Mq.logits(np.array([h+G])),T)[0];x=int(r.choice(len(q),p=q));G.append(x);Q.append(q)
    P=[]
    batch=np.array([(h+G[:i])[-Mp.ctx:] for i in range(gamma+1)]);lg=Mp.logits(batch);P=prob(lg,T);n=gamma
    for i in range(gamma):
        rr=r.random();pi=P[i][G[i]];qi=lenience*Q[i][G[i]]
        if rr>(pi/qi if qi>0 else 1):n=i;break
    if n<gamma:
        a=np.maximum(0,P[n]-lenience*Q[n]);s=a.sum();a=a/s if s>1e-12 else P[n]
    else:a=P[n]
    t=int(r.choice(len(a),p=a/a.sum()));return G[:n]+[t],n
def alpha(Mp,Mq,seed,T,start,lenience=1.,N=3000):
    r=np.random.default_rng(seed);hist=np.array([list(start)]);tot=[];
    for _ in range(N):
        p=prob(Mp.logits(hist),T)[0];q=prob(Mq.logits(hist),T)[0];tot.append(float((np.minimum(p,q*lenience)).sum()/lenience));x=int(r.choice(len(p),p=p));hist=np.c_[hist[:,1:],[[x]]]
    return float(np.mean(tot))
def fidelity(Mp,Mq,seed,start,gamma=3,T=1.,n=6000,L=3):
    """同一前缀，生成 L 个字符。比较直接采样和推测解码的联合分布距离，用两次直接采样当噪声底"""
    r=np.random.default_rng(seed)
    def direct(n):
        x=np.tile(start,(n,1))
        for _ in range(L):
            pr=prob(Mp.logits(x),T);c=(pr.cumsum(1)>r.random((n,1))).argmax(1);x=np.c_[x,c]
        return x[:,-L:]
    def specgen(n,cheat=False):
        o=[]
        for _ in range(n):
            h=list(start);g=[]
            while len(g)<L:
                ne,_=spec(Mp,Mq,h+g,gamma,T,r);g+=ne
            o.append(g[:L])
        return np.array(o)
    def naive(n):
        """错误做法：小模型猜的，只要 p(x)>0 就接受（一路用小模型）"""
        x=np.tile(start,(n,1))
        for _ in range(L):
            pr=prob(Mq.logits(x),T);c=(pr.cumsum(1)>r.random((n,1))).argmax(1);x=np.c_[x,c]
        return x[:,-L:]
    V=Mp.V;key=lambda a:(a*V**np.arange(L)).sum(1);tv=lambda a,b:float(.5*np.abs(np.bincount(np.searchsorted(u,key(a)),minlength=len(u))/len(a)-np.bincount(np.searchsorted(u,key(b)),minlength=len(u))/len(b)).sum())
    A,B,S,Nv=direct(n),direct(n),specgen(n),naive(n);u=np.unique(np.r_[key(A),key(B),key(S),key(Nv)]);return dict(noise=tv(A,B),spec=tv(A,S),small=tv(A,Nv))
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 15000;t0=time.time();T_=corpus();ch=sorted(set(T_));cid={c:i for i,c in enumerate(ch)};D=np.array([cid[c] for c in T_],np.int32);V=len(ch);n=len(D);r=np.random.default_rng(sd);idx=np.arange(40,n);r.shuffle(idx);tr=idx[:n-60000];res={'vocab':''.join(ch)}
    Mp=LM(V,16,12,256,sd);train(Mp,D,tr,S,sd+1);print('Mp',round(time.time()-t0),flush=True)
    Ms=LM(V,6,8,32,sd+5);train(Ms,D,tr,S//2,sd+6);Mm=LM(V,10,10,96,sd+7);train(Mm,D,tr,S//2,sd+8);print('Mq',round(time.time()-t0),flush=True)
    Qs=dict(uni=Gram(D,V,1),bi=Gram(D,V,2),small=Ms,mid=Mm);res['flops']={k:q.flops/Mp.flops for k,q in Qs.items()};res['params']={'p':Mp.flops,'small':Ms.flops,'mid':Mm.flops}
    start=np.array([cid[c] for c in 'the model is trained on '[-16:]]);res['alpha']={};
    for T in (0,.7,1.):
        res['alpha'][str(T)]={k:alpha(Mp,q,sd+20,T,start) for k,q in Qs.items()};print(T,res['alpha'][str(T)],round(time.time()-t0),flush=True)
    res['gamma']={}
    for T in (0,1.):
        row={}
        for g in (1,2,3,5,8,12):
            rr=np.random.default_rng(sd+30);h=list(start);c=0;acc=0;tk=[]
            while len(tk)<4000:
                ne,a=spec(Mp,Ms,(h+tk)[-16:],g,T,rr);tk+=ne;c+=1;acc+=a
            row[str(g)]=dict(tokens=len(tk)/c,calls=c);print(T,g,round(len(tk)/c,3),round(time.time()-t0),flush=True)
        res['gamma'][str(T)]=row
    res['fid']=fidelity(Mp,Ms,sd+40,start);print(res['fid'],round(time.time()-t0),flush=True)
    # 宽容度
    res['len']={str(l):alpha(Mp,Ms,sd+21,1.,start,lenience=l) for l in (1.,.5,.3,.1)};print(res['len'],flush=True)
    if sd==0:
        np.savez('m_p.npz',E=Mp.E,**{f'W{i}':w for i,w in enumerate(Mp.W)},**{f'b{i}':b for i,b in enumerate(Mp.b)});np.savez('m_q.npz',E=Ms.E,**{f'W{i}':w for i,w in enumerate(Ms.W)},**{f'b{i}':b for i,b in enumerate(Ms.b)})
    res['sec']=time.time()-t0;json.dump(res,open(f'res{sd}.json','w'));print('done',round(time.time()-t0),flush=True)
