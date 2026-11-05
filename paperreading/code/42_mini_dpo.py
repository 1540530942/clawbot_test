#!/usr/bin/env python3
# 第 42 期 DPO 玩具实验：表格推导验证和迷你语言模型上的偏好对齐
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

"""第 42 期玩具实验：DPO。第一部分是有精确答案的表格实验：回答只有 K 种，π* 可以直接算，看 DPO 是否走到 π*。
第二部分是迷你语言模型：续写影评，真实奖励是情感函数（含“不”字翻转）。比较 DPO、PPO（学到的奖励和真实奖励）、只学好回答、Unlikelihood。
和论文一样报告奖励-KL 前沿。这是玩具。"""
import sys,json,time,numpy as np
POS=['好','棒','爱','赞','美'];NEG=['差','烂','恨','糟','丑'];NEU=['这部','电影','剧情','演员','音乐','画面','的','很','真'];vocab=['不']+POS+NEG+NEU;ID={c:i for i,c in enumerate(vocab)};V=len(vocab);L=8;PL=2;NOT=0
SENT=np.zeros(V);SENT[[ID[c] for c in POS]]=1;SENT[[ID[c] for c in NEG]]=-1
def reward(y):
    """真实奖励：情感词加减分，紧跟在“不”后面的情感词翻转。y:(B,L)"""
    s=SENT[y].copy();s[:,1:]*=np.where(y[:,:-1]==NOT,-1,1);return s.sum(1)
def corpus(r,n):
    """参考语料：语法简单的句子，情感随机，偏负面一点"""
    S=np.zeros((n,L),int);subj=[ID[c] for c in NEU[:6]];ad=[ID['很'],ID['真'],NOT,ID['的']];
    for i in range(n):
        t=[];
        while len(t)<L:
            t.append(int(r.choice(subj)));t.append(int(r.choice(ad)));c=POS if r.random()<.4 else NEG;t.append(ID[c[int(r.integers(5))]])
        S[i]=t[:L]
    return S
def make(seed):
    return Enc(V,L,d=32,h=4,ff=64,N=2,seed=seed,causal=True,pre=True)
def clone(w):
    m=make(0)
    for p,v in zip(m.ps,w):p.v=v.copy()
    return m
def get(m):return [p.v.copy() for p in m.ps]
def lp_all(m,y):
    """每条序列（提示之后）对数概率（节点）与逐 token 分布。返回 p 节点、log p 值"""
    lg=m.lm(y[:,:-1]);p=lg.softmax();return p
def logp(pv,y):
    idx=np.indices(y[:,1:].shape);return np.log(pv[(*idx,y[:,1:])]+1e-12)*(np.arange(1,L)>=PL)
def seed_g(pv,y,c):
    """对每条序列 logπ 的梯度种子，乘系数 c（B,）"""
    idx=np.indices(y[:,1:].shape);g=np.zeros(pv.shape);g[(*idx,y[:,1:])]=(c[:,None]/(pv[(*idx,y[:,1:])]+1e-12))*(np.arange(1,L)>=PL);return g
def sample(m,P,r,temp=1.):
    x=P.copy()
    for t in range(L-PL):
        lg=m.lm(x).v[:,-1]/temp;pr=np.exp(lg-lg.max(1,keepdims=True));pr/=pr.sum(1,keepdims=True);a=(pr.cumsum(1)>r.random((len(x),1))).argmax(1);x=np.c_[x,a]
    return x
def seqlp(m,y):
    return logp(m.lm(y[:,:-1]).softmax().v,y).sum(1)
def prompts(r,n):
    subj=[ID[c] for c in NEU[:6]];ad=[ID['很'],ID['真'],NOT,ID['的']];return np.c_[r.choice(subj,n),r.choice(ad,n)]
def evaluate(m,ref,r,n=400):
    y=sample(m,prompts(r,n),r);a=seqlp(m,y);b=seqlp(ref,y);uniq=len({tuple(q) for q in y[:,PL:]});top=np.bincount(y[:,PL:].ravel(),minlength=V).max()/y[:,PL:].size
    return dict(rw=float(reward(y).mean()),kl=float((a-b).mean()),uniq=uniq/n,top=float(top))
def pairs(ref,r,N,mode='det',k=4):
    """每个提示采样 k 条回答，两两成对。det：真实奖励高的赢；bt：按 σ(Δ) 随机。平局丢弃"""
    W,Lo=[],[];n=int(np.ceil(N/(k*(k-1)/2)))
    P=np.repeat(prompts(r,n),k,0);Y=sample(ref,P,r);R=reward(Y)
    for i in range(n):
        for a in range(k):
            for b in range(a+1,k):
                u,v=i*k+a,i*k+b
                if mode=='bt':
                    w=u if r.random()<1/(1+np.exp(R[v]-R[u])) else v;l=u+v-w
                    W.append(Y[w]);Lo.append(Y[l])
                elif R[u]!=R[v]:
                    w,l=(u,v) if R[u]>R[v] else (v,u);W.append(Y[w]);Lo.append(Y[l])
    return np.array(W),np.array(Lo)
def train_pref(m,ref,W,Lo,steps,kind,beta,r,rec,lr=5e-4,bs=32):
    o=Adam(m.ps);hist=[]
    for st in range(1,steps+1):
        i=r.integers(0,len(W),bs);a,b=W[i],Lo[i];
        pa=m.lm(a[:,:-1]).softmax();la=logp(pa.v,a).sum(1)
        if kind=='pref':
            pa.backward(-seed_g(pa.v,a,np.ones(bs)/bs))
        else:
            pb=m.lm(b[:,:-1]).softmax();lb=logp(pb.v,b).sum(1)
            if kind=='dpo':
                ra,rb=seqlp(ref,a),seqlp(ref,b);u=beta*((la-ra)-(lb-rb));wt=beta/(1+np.exp(np.clip(u,-30,30)));ca,cb=-wt/bs,wt/bs
            elif kind=='ul':
                ca,cb=-np.ones(bs)/bs,beta*np.ones(bs)/bs
            root=T(0.,(pa,pb));root.bk=lambda g,sa=seed_g(pa.v,a,ca),sb=seed_g(pb.v,b,cb):(sa,sb);root.backward()
        for p in m.ps:
            if p.g is not None:np.clip(p.g,-1,1,p.g)
        o.step(lr)
        if st%rec==0:hist.append(evaluate(m,ref,r,200))
    return hist
def fit_rm(W,Lo,r,steps=3000):
    """奖励模型：一元和二元词袋，Bradley-Terry 训练"""
    def F(y):
        x=np.zeros((len(y),V+V*V));
        for i,s in enumerate(y[:,PL:]):
            np.add.at(x[i],s,1);np.add.at(x[i],V+s[:-1]*V+s[1:],1)
        return x
    A,B=F(W),F(Lo);w=np.zeros(A.shape[1]);D=A-B
    for st in range(steps):
        i=r.integers(0,len(D),256);p=1/(1+np.exp(np.clip(D[i]@w,-30,30)));w+=.05*(D[i].T@p/256-1e-3*w)
    return lambda y:F(y)@w
def train_ppo(m,ref,rf,steps,beta,r,rec,lr=5e-4,bs=64,clip=.2,ep=2):
    """PPO：回报 r - β(logπ-logπ_ref)，减去批均值作基线（相当于论文说的归一化），序列级裁剪"""
    o=Adam(m.ps);hist=[]
    for st in range(1,steps+1):
        y=sample(m,prompts(r,bs),r);old=seqlp(m,y);R=rf(y)-beta*(old-seqlp(ref,y));Ad=(R-R.mean())/(R.std()+1e-6)
        for _ in range(ep):
            p=m.lm(y[:,:-1]).softmax();n=logp(p.v,y).sum(1);rt=np.exp(np.clip(n-old,-5,5));act=(np.abs(rt-1)<clip)|((rt-1)*Ad<0);p.backward(seed_g(p.v,y,-Ad*rt*act/bs))
            for q in m.ps:
                if q.g is not None:np.clip(q.g,-1,1,q.g)
            o.step(lr)
        if st%rec==0:hist.append(evaluate(m,ref,r,200))
    return hist
def tabular(sd):
    """K 种回答。π_ref、r* 随机。全批量优化 DPO，和解析解比较。deterministic=True 时标签确定"""
    r=np.random.default_rng(sd+50);K=16;ref=r.dirichlet(np.ones(K)*2);rs=r.normal(0,1,K);out={};sig=lambda z:1/(1+np.exp(np.clip(z,-40,40)))
    for beta in (.5,1.,2.,5.):
        ana=ref*np.exp(rs/beta);ana/=ana.sum();row={}
        for det in (False,True):
            th=np.log(ref).copy();kls=[]
            Pw=np.outer(ref,ref)*(sig(rs[None,:]-rs[:,None]) if not det else (rs[:,None]>rs[None,:])*1.)
            for st in range(1,40001):
                pi=np.exp(th);pi/=pi.sum();u=np.log(pi/ref);M=beta*(u[:,None]-u[None,:]);wt=sig(M)*Pw;g=beta*(wt.sum(1)-wt.sum(0));gr=g-(g.sum())*pi;th+=.5*gr;th-=th.mean()
                if st in(1000,10000,40000):kls.append(float((pi*np.log(pi/ref)).sum()))
            row['det' if det else 'bt']=dict(tv=float(.5*abs(pi-ana).sum()),kl=kls,pi=np.round(pi,4).tolist())
        row['ana_kl']=float((ana*np.log(ana/ref)).sum());row['ana']=np.round(ana,4).tolist();out[str(beta)]=row
    out['ref']=np.round(ref,4).tolist();out['rs']=np.round(rs,3).tolist();return out
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 1500;t0=time.time();res={}
    res['tab']=tabular(sd);print('tab',{k:(v['bt']['tv'],v['det']['kl'],v['ana_kl']) for k,v in res['tab'].items() if k not in('ref','rs')},round(time.time()-t0),flush=True);json.dump(res,open(f'res{sd}.json','w'),ensure_ascii=False)
    if S<=0:sys.exit()
    r=np.random.default_rng(sd);C=corpus(r,5000);m=make(sd);o=Adam(m.ps)
    for st in range(1,2001):
        x=C[r.integers(0,len(C),32)];v,p,g=ce(m.lm(x[:,:-1]),x[:,1:]);p.backward(g);o.step(2e-3*(1 if st<1600 else .2))
    print('sft',v,round(time.time()-t0),flush=True);w0=get(m);ref=clone(w0);res['ref']=evaluate(ref,ref,r,1000);print(res['ref'],flush=True)
    W,Lo=pairs(ref,r,6000);res['npairs']=len(W);print('pairs',len(W),round(time.time()-t0),flush=True);rf=fit_rm(W,Lo,r);yt=sample(ref,prompts(r,1000),r);res['rm_corr']=float(np.corrcoef(rf(yt),reward(yt))[0,1]);print('rm',res['rm_corr'],flush=True)
    rec=max(1,S//6);arms=[('pref','pref',0),('ul.1','ul',.1),('ul.5','ul',.5),('ul1','ul',1.)]+[(f'dpo{b}','dpo',b) for b in (.1,.25,.5,1.,5.)]
    for nm,k,b in arms:
        ms=clone(w0);res[nm]=train_pref(ms,ref,W,Lo,S,k,b,np.random.default_rng(sd+11),rec);print(sd,nm,res[nm][-1],round(time.time()-t0),flush=True);json.dump(res,open(f'res{sd}.json','w'),ensure_ascii=False)
        if sd==0 and nm=='dpo0.5':np.savez('m_dpo.npz',**{f'p{i}':p.v for i,p in enumerate(ms.ps)})
    if sd==0:np.savez('m_ref.npz',**{f'p{i}':p.v for i,p in enumerate(ref.ps)})
    PS=S//2
    for nm,rw,b in (('ppo.1',rf,.1),('ppo.5',rf,.5),('ppo2',rf,2.),('gt.1',reward,.1),('gt.5',reward,.5),('gt2',reward,2.)):
        ms=clone(w0);res[nm]=train_ppo(ms,ref,rw,PS,b,np.random.default_rng(sd+12),max(1,PS//6));print(sd,nm,res[nm][-1],round(time.time()-t0),flush=True);json.dump(res,open(f'res{sd}.json','w'),ensure_ascii=False)
    # 数据量和带噪标签
    W2,L2=W[:300],Lo[:300];ms=clone(w0);res['dpo_small']=train_pref(ms,ref,W2,L2,S,'dpo',.5,np.random.default_rng(sd+11),rec)
    W3,L3=pairs(ref,r,6000,'bt');ms=clone(w0);res['dpo_bt']=train_pref(ms,ref,W3,L3,S,'dpo',.5,np.random.default_rng(sd+11),rec);print(sd,'small,bt',res['dpo_small'][-1],res['dpo_bt'][-1],flush=True)
    res['sec']=time.time()-t0;json.dump(res,open(f'res{sd}.json','w'),ensure_ascii=False);print('done',round(time.time()-t0),flush=True)
