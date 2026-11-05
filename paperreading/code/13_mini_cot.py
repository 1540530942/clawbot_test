#!/usr/bin/env python3
# 第 13 期迷你实验：连加连减（模 10）上比较直接答、写步骤、写点、答案在前。
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

"""第 13 期玩具实验：连加连减（模 10）。同一道题四种答题格式：
direct 直接给答案；cot 先写每一步的中间结果；dots 写同样多的点（只多算力，不写内容）；
after 先给答案、再写步骤。还试了不同模型大小、更长的算式和“篡改中间步”。
注意：这里是把格式写进训练数据（类似 Ling 2017、Cobbe 2021），不是论文那种少样本提示。"""
import sys,json,time,numpy as np
vocab=list('0123456789')+['+','-','=',';','.','<e>','<p>']
ID={c:i for i,c in enumerate(vocab)};PLUS,MINUS,EQ,SEMI,DOT,EOS,PAD=[ID[c] for c in('+','-','=',';','.','<e>','<p>')]
L=44;V=len(vocab)
def sinus(L,d):
    p=np.arange(L)[:,None];t=10000.0**(-2*np.arange(d//2+1)/d);s=np.zeros((L,d));s[:,0::2]=np.sin(p*t[:(d+1)//2]);s[:,1::2]=np.cos(p*t[:d//2]);return s
def problem(r,K):
    ds=r.integers(0,10,K);ops=r.integers(0,2,K-1);pr=[int(ds[0])];run=[];c=int(ds[0])
    for o,d in zip(ops,ds[1:]):
        pr+=[PLUS if o==0 else MINUS,int(d)];c=(c+int(d))%10 if o==0 else (c-int(d))%10;run.append(c)
    return pr+[EQ],run,c
def answer(fmt,run,a):
    if fmt=='direct':return [a,EOS]
    if fmt=='cot':return run+[SEMI,a,EOS]
    if fmt=='dots':return [DOT]*len(run)+[SEMI,a,EOS]
    if fmt=='after':return [a,SEMI]+run+[EOS]
def batch(r,B,fmt,Ks):
    S=[];
    for _ in range(B):
        p,run,a=problem(r,int(r.choice(Ks)));S.append((p,answer(fmt,run,a)))
    n=max(len(p)+len(q) for p,q in S)
    x=np.full((B,n-1),PAD);y=np.zeros((B,n-1),int);w=np.zeros((B,n-1))
    for i,(p,q) in enumerate(S):
        s=p+q;x[i,:len(s)-1]=s[:-1];y[i,:len(s)-1]=s[1:];w[i,len(p)-1:len(s)-1]=1
    return x,y,w
def make(d,h,ff,seed):
    m=Enc(V,L,d=d,h=h,ff=ff,N=2,seed=seed,causal=True);m.P.v=sinus(L,d);m.tr=[p for p in m.ps if p is not m.P];return m
def train(fmt,d,h,ff,seed,steps,Ks=(2,3,4,5)):
    m=make(d,h,ff,seed);o=Adam(m.tr);r=np.random.default_rng(seed+11);hist=[];t=time.time()
    for st in range(1,steps+1):
        x,y,w=batch(r,32,fmt,Ks);v,p,g=ce(m.lm(x),y,w);p.backward(g);m.P.g=None
        o.step(min(2e-3,2e-3*st/150)*(.3 if st>.8*steps else 1))
        if st%max(1,steps//12)==0:hist.append((st,round(float(v),3)))
    return m,hist,round(time.time()-t,1)
def decode(m,pr,maxn,force=None):
    """贪心生成。pr:同长提示；force:{位置:数组}强行改第几个生成词"""
    x=pr.copy();B=len(x)
    for t in range(maxn):
        nx=m.lm(x).v[:,-1].argmax(-1)
        if force and t in force:nx=force[t]
        x=np.c_[x,nx]
    return x[:,pr.shape[1]:]
def parse(fmt,g):
    if fmt in('direct','after'):return g[:,0]
    o=np.full(len(g),-1)
    for i,row in enumerate(g):
        j=np.where(row==SEMI)[0]
        if len(j) and j[0]+1<len(row):o[i]=row[j[0]+1]
    return o
def score(m,fmt,K,seed,B=200):
    r=np.random.default_rng(seed);P=[];A=[]
    for _ in range(B):
        p,run,a=problem(r,K);P.append(p);A.append(a)
    g=decode(m,np.array(P),K+2);return float((parse(fmt,g)==np.array(A)).mean())
def tamper(m,K,seed,B=200):
    """把第一步中间结果改错，答案跟着错链走，还是仍给真答案？"""
    r=np.random.default_rng(seed);P=[];A=[];Pr=[]
    for _ in range(B):
        p,run,a=problem(r,K);P.append(p);A.append(a);Pr.append((run[0],a))
    fa=np.array([(u+1+r.integers(0,9))%10 for u,_ in Pr]);P=np.array(P);A=np.array(A);f0=np.array([u for u,_ in Pr])
    g=decode(m,P,K+2,force={0:fa});out=parse('cot',g);delta=(fa-f0)%10
    return dict(follow=float((out==(A+delta)%10).mean()),truth=float((out==A).mean()))
if __name__=='__main__':
    sd=int(sys.argv[1]);steps=int(sys.argv[2]) if len(sys.argv)>2 else 1200
    if steps<=0:
        m=make(16,2,32,0);x,y,w=batch(np.random.default_rng(0),2,'cot',(2,3));v,p,g=ce(m.lm(x),y,w);p.backward(g);P=m.layers[0]['q'];a=P.g[1,2];e=1e-5;P.v[1,2]+=e;v1=ce(m.lm(x),y,w)[0];P.v[1,2]-=2*e;v2=ce(m.lm(x),y,w)[0];P.v[1,2]+=e;print('grad',abs(a-(v1-v2)/2/e));sys.exit()
    out={}
    runs=[('direct',48,4,96),('cot',48,4,96),('dots',48,4,96),('after',48,4,96),('direct',24,4,48),('cot',24,4,48),('direct',64,4,128),('cot',64,4,128)]
    for fmt,d,h,ff in runs:
        m,hist,sec=train(fmt,d,h,ff,sd,steps);res=dict(hist=hist,sec=sec,acc={K:score(m,fmt,K,900+sd+K) for K in(2,3,4,5,7,9)})
        if fmt=='cot' and d==48:
            res['tamper']={K:tamper(m,K,800+sd+K) for K in(3,5)}
        out[f'{fmt}/{d}']=res;print(sd,fmt,d,res['acc'],sec,res.get('tamper'),flush=True)
        json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
        if sd==0 and d==48 and fmt in('direct','cot'):
            np.savez(f'{fmt}.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)})
