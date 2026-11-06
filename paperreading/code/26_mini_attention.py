#!/usr/bin/env python3
# 第 26 期迷你注意力翻译：双向 GRU、加性对齐、maxout，对比定长向量和注意力，并量化对齐准确率
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

"""第 26 期玩具实验：GRU 编码解码（定长向量）和注意力（RNNsearch）。
玩具语言同第 25 期：主动句 cat chases dog 和被动句 dog chased_by cat 都译成 猫 追 狗。
分两种训练长度（最多 2 个分句、最多 4 个分句），测试到 8 个分句。这是玩具，不代表论文结果。"""
import sys,json,time,math,collections,numpy as np
NOUN=['cat','dog','fish','bird','horse','cow','pig','sheep','wolf','fox','bear','lion']
NZH=list('猫狗鱼鸟马牛猪羊狼狐熊狮')
VERB=['chases','bites','sees','helps','pushes','pulls']
VZH=list('追咬看帮推拉')
EOS=0
SRC=['<e>']+NOUN+VERB+[v+'_by' for v in VERB]+['and'];TGT=['<e>']+NZH+VZH+['，']
SI={w:i for i,w in enumerate(SRC)};TI={w:i for i,w in enumerate(TGT)}
def clause(r):
    s,o=r.choice(12,2,replace=False);v=int(r.integers(0,6));a=r.random()<.5
    return ([NOUN[s],VERB[v],NOUN[o]] if a else [NOUN[o],VERB[v]+'_by',NOUN[s]]),[NZH[s],VZH[v],NZH[o]],a
def sentence(r,K):
    S=[];Tt=[];al=[];off=0
    for i in range(K):
        if i:S.append('and');Tt.append('，');al.append(off);off+=1
        a,b,act=clause(r);S+=a;Tt+=b;al+=[off,off+1,off+2] if act else [off+2,off+1,off];off+=3
    return S,Tt,al
def batch(r,B,K):
    P=[sentence(r,K) for _ in range(B)];X=np.array([[SI[w] for w in s]+[EOS] for s,_,_ in P]);Y=np.array([[TI[w] for w in t]+[EOS] for _,t,_ in P]);return X,Y,np.array([a+[len(a)+0] for _,_,a in P])
sg=lambda z:1/(1+np.exp(-np.clip(z,-30,30)))
def sigm(x):
    p=sg(x.v);r=T(p,(x,));r.bk=lambda g:(g*p*(1-p),);return r
def tanh(x):
    p=np.tanh(x.v);r=T(p,(x,));r.bk=lambda g:(g*(1-p*p),);return r
def cat(a,b):
    r=T(np.concatenate([a.v,b.v],-1),(a,b));n=a.v.shape[-1];r.bk=lambda g:(g[...,:n],g[...,n:]);return r
def stack(xs):
    r=T(np.stack([x.v for x in xs],1),tuple(xs));r.bk=lambda g:tuple(g[:,i] for i in range(len(xs)));return r
def pick(E,ids):
    r=T(E.v[ids],(E,))
    def bk(g):
        o=np.zeros_like(E.v);np.add.at(o,ids,g);return (o,)
    r.bk=bk;return r
def maxout(x):
    B=x.v.shape;a=x.v.reshape(B[0],-1,2);i=a.argmax(-1);o=np.take_along_axis(a,i[...,None],-1)[...,0];r=T(o,(x,))
    def bk(g):
        z=np.zeros_like(a);np.put_along_axis(z,i[...,None],g[...,None],-1);return (z.reshape(B),)
    r.bk=bk;return r
class M:
    def __init__(s,n,m,seed,att):
        r=np.random.default_rng(seed);s.n,s.m,s.att=n,m,att;s.ps=[];s.names=[];nn=n;l=n//2
        def G(a,b,sd=.01,name=''):
            t=T(r.normal(0,sd,(a,b)));s.ps.append(t);s.names.append(name);return t
        def O(a,name):
            q,_=np.linalg.qr(r.normal(0,1,(a,a)));t=T(q);s.ps.append(t);s.names.append(name);return t
        def gru(pre,ctx):
            d=dict(W=G(m,n,name=pre+'W'),Wz=G(m,n,name=pre+'Wz'),Wr=G(m,n,name=pre+'Wr'),U=O(n,pre+'U'),Uz=O(n,pre+'Uz'),Ur=O(n,pre+'Ur'))
            if ctx:d.update(C=G(ctx,n,name=pre+'C'),Cz=G(ctx,n,name=pre+'Cz'),Cr=G(ctx,n,name=pre+'Cr'))
            return d
        s.Es=G(len(SRC),m,name='Es');s.Et=G(len(TGT),m,name='Et');ctx=2*n if att else n
        s.f=gru('f',0);s.dec=gru('d',ctx);s.Ws=G(n,n,name='Ws')
        if att:
            s.b=gru('b',0);s.Wa=G(n,n,.001,'Wa');s.Ua=G(2*n,n,.001,'Ua');s.va=T(np.zeros((n,1)));s.ps.append(s.va);s.names.append('va')
        s.Uo=G(n,2*l,name='Uo');s.Vo=G(m,2*l,name='Vo');s.Co=G(ctx,2*l,name='Co');s.Wo=G(l,len(TGT),name='Wo');s.ctx=ctx
    def gstep(s,g,e,h,c=None):
        z=e@g['Wz']+h@g['Uz'];q=e@g['Wr']+h@g['Ur']
        if c is not None:z=z+c@g['Cz'];q=q+c@g['Cr']
        z=sigm(z);q=sigm(q);t=e@g['W']+(q*h)@g['U']
        if c is not None:t=t+c@g['C']
        return (z*-1+1.)*h+z*tanh(t)
    def encode(s,X):
        B,Tx=X.shape;h=T(np.zeros((B,s.n)));F=[]
        for t in range(Tx):
            h=s.gstep(s.f,pick(s.Es,X[:,t]),h);F.append(h)
        if not s.att:return h,None,None,tanh(h@s.Ws)
        h=T(np.zeros((B,s.n)));R=[None]*Tx
        for t in range(Tx-1,-1,-1):
            h=s.gstep(s.b,pick(s.Es,X[:,t]),h);R[t]=h
        H=stack([cat(a,b) for a,b in zip(F,R)]);return None,H,H@s.Ua,tanh(R[0]@s.Ws)
    def dstep(s,y,st,c0,H,UH):
        B=len(y);e=pick(s.Et,y);al=None
        if s.att:
            sc=tanh((st@s.Wa).rs(B,1,s.n)+UH)@s.va;a=sc.rs(B,H.v.shape[1]).softmax();al=a.v;c=(a.rs(B,1,H.v.shape[1])@H).rs(B,2*s.n)
        else:c=c0
        out=maxout(st@s.Uo+e@s.Vo+c@s.Co)@s.Wo;new=s.gstep(s.dec,e,st,c);return out,new,al
    def loss(s,X,Y):
        c0,H,UH,st=s.encode(X);B=len(X);y=np.zeros(B,int);tot=0.;Ls=[]
        for t in range(Y.shape[1]):
            out,st,_=s.dstep(y,st,c0,H,UH);p=out.softmax();py=p.v[np.arange(B),Y[:,t]];tot-=np.log(py+1e-12).sum();sd=np.zeros(p.v.shape);sd[np.arange(B),Y[:,t]]=-1/(py+1e-12)/B;Ls.append((p,sd));y=Y[:,t]
        # 所有输出节点共用一张图，先把种子写好再统一反传
        return tot/B,Ls
    def grad(s,Ls):
        root=T(0.,tuple(p for p,_ in Ls));seeds=[g for _,g in Ls];root.bk=lambda g:tuple(seeds);root.backward()
    def greedy(s,X,maxn):
        c0,H,UH,st=s.encode(X);B=len(X);y=np.zeros(B,int);o=[];As=[]
        for t in range(maxn):
            out,st,al=s.dstep(y,st,c0,H,UH);y=out.v.argmax(-1);o.append(y);As.append(al)
        return np.array(o).T,As
def strip(row):
    o=[]
    for t in row:
        if t==EOS:break
        o.append(int(t))
    return o
def bleu(hyps,refs):
    pn_=[];hl=sum(map(len,hyps));rl=sum(map(len,refs))
    for n in range(1,5):
        m=t=0
        for h,r in zip(hyps,refs):
            ch=collections.Counter(tuple(h[i:i+n]) for i in range(len(h)-n+1));cr=collections.Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1));m+=sum(min(v,cr[k]) for k,v in ch.items());t+=sum(ch.values())
        pn_.append(m/t if t else 0)
    if min(pn_)==0:return 0.
    bp=1 if hl>rl else math.exp(1-rl/max(hl,1));return 100*bp*math.exp(sum(math.log(p) for p in pn_)/4)
def evaluate(m,seed,Ks=(1,2,3,4,5,6,8),n=60):
    r=np.random.default_rng(seed);out={};ali=[]
    for K in Ks:
        X,Y,al=batch(r,n,K);o,As=m.greedy(X,Y.shape[1]+3);H=[strip(a) for a in o];R=[strip(a) for a in Y];out[K]=dict(bleu=bleu(H,R),exact=float(np.mean([a==b for a,b in zip(H,R)])))
        if m.att:
            A=np.array(As[:Y.shape[1]-1]).transpose(1,0,2);ok=[];ok1=[];okc=[]
            for i in range(n):
                if H[i]==R[i]:
                    g=A[i].argmax(-1);t=al[i][:Y.shape[1]-1];ok.append((g==t).mean());ok1.append((abs(g-t)<=1).mean());okc.append((g//4==t//4).mean())
            out[K]['align']=float(np.mean(ok)) if ok else None;out[K]['align1']=float(np.mean(ok1)) if ok else None;out[K]['align_clause']=float(np.mean(okc)) if ok else None
    return out
def train(m,Ks,steps,seed,lr=2e-3,B=32):
    r=np.random.default_rng(seed+50);o=Adam(m.ps);h=[]
    for st in range(1,steps+1):
        X,Y,_=batch(r,B,int(r.choice(Ks)));L,Ls=m.loss(X,Y);m.grad(Ls);n=math.sqrt(sum((p.g*p.g).sum() for p in m.ps if p.g is not None))
        if n>1:
            for p in m.ps:
                if p.g is not None:p.g=p.g/n
        o.step(lr*(1 if st<.8*steps else .3))
        if st%max(1,steps//20)==0:h.append((st,round(L/Y.shape[1],3)))
    return h
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 3000;n=int(sys.argv[3]) if len(sys.argv)>3 else 48
    if S<=0:
        m=M(8,6,0,True);r=np.random.default_rng(0);X,Y,_=batch(r,3,2);L,Ls=m.loss(X,Y);m.grad(Ls);bad=0
        for p in m.ps:
            g=p.g.copy();i=np.unravel_index(int(np.argmax(abs(g))),g.shape);e=1e-5;p.v[i]+=e;a=m.loss(X,Y)[0];p.v[i]-=2*e;b=m.loss(X,Y)[0];p.v[i]+=e;bad=max(bad,abs((a-b)/2/e-g[i])/(abs(g[i])+1e-9))
        print('最大相对误差',bad);sys.exit()
    out={};t0=time.time()
    for k,att,Ks in (('enc2',False,(1,2)),('att2',True,(1,2)),('enc4',False,(1,2,3,4)),('att4',True,(1,2,3,4))):
        m=M(n,32,sd,att);h=train(m,Ks,S,sd);res=evaluate(m,8000+sd);out[k]=dict(hist=h,res=res,sec=round(time.time()-t0));print(sd,k,h[-1],{K:(round(v['bleu'],1),round(v['exact'],2),v.get('align')) for K,v in res.items()},out[k]['sec'],flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
        if sd==0 and k=='att4':np.savez('m_att.npz',**{nm:p.v for nm,p in zip(m.names,m.ps)})
