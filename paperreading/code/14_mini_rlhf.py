#!/usr/bin/env python3
# 第 14 期迷你实验：预训练、SFT、奖励模型和 PPO，走完整条 InstructGPT 流程。
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

"""第 14 期玩具实验：完整走一遍 InstructGPT。
“互联网”语料里，指令后面常常接另一条指令（模型学会的是续写而不是回答）。
三步：少量人工示范微调（SFT）；对输出排序训练奖励模型（RM）；用 RM+KL 惩罚做 PPO。
真实偏好是我们写的函数，人工标注带噪声。这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
LET=list('abcdef');vocab=['<p>','<e>','=>','重复','倒序','最大','下一个','坏词','也许']+LET
ID={c:i for i,c in enumerate(vocab)};PAD,EOS,ARR=0,1,2;REP,REV,MAX,NXT,BAD,HEDGE=3,4,5,6,7,8;A0=9;V=len(vocab);L=14;LEN=3;MAXR=6
def sinus(L,d):
    p=np.arange(L)[:,None];t=10000.0**(-2*np.arange(d//2)/d);s=np.zeros((L,d));s[:,0::2]=np.sin(p*t);s[:,1::2]=np.cos(p*t);return s
def truth(op,a):
    return list(a) if op==REP else list(a[::-1]) if op==REV else [max(a)]
def prompt(r,op=None):
    op=op or int(r.choice([REP,REV,MAX]));a=[int(x) for x in A0+r.integers(0,6,LEN)];return [op]+a+[ARR],op,a
def internet(r):
    """预训练语料：60% 接另一条指令，15% 正确，10% 带坏词，5% 先说也许，10% 下一个字母（公开任务）"""
    u=r.random()
    if u<.10:
        x=int(r.integers(0,5));return [NXT,A0+x,ARR,A0+x+1,EOS]
    p,op,a=prompt(r)
    if u<.70:
        q,_,b=prompt(r);return p+q[:4]+[EOS]
    if u<.85:return p+truth(op,a)+[EOS]
    if u<.90:return p+[HEDGE]+truth(op,a)+[EOS]
    return p+[BAD]+truth(op,a)+[EOS]
def pad(S,n=None):
    n=n or max(map(len,S));x=np.full((len(S),n),PAD)
    for i,s in enumerate(S):x[i,:len(s)]=s
    return x
def lmbatch(S):
    x=pad(S);return x[:,:-1],x[:,1:],(x[:,1:]!=PAD).astype(float)
def make(d,h,ff,seed):
    m=Enc(V,L,d=d,h=h,ff=ff,N=2,seed=seed,causal=True);m.P.v=sinus(L,d);m.tr=[p for p in m.ps if p is not m.P];return m
def clone(w,d,h,ff):
    m=make(d,h,ff,0)
    for p,v in zip(m.ps,w):p.v=v.copy()
    return m
def get(m):return [p.v.copy() for p in m.ps]
def pretrain(m,r,steps):
    o=Adam(m.tr)
    for st in range(1,steps+1):
        x,y,w=lmbatch([internet(r) for _ in range(32)]);v,p,g=ce(m.lm(x),y,w);p.backward(g);m.P.g=None;o.step(min(2e-3,2e-3*st/200)*(.3 if st>.8*steps else 1))
def sample(m,P,r,temp=1.,greedy=False):
    """P:提示数组（同长）。返回 回答 列表、每步对数概率"""
    B=len(P);x=P.copy();done=np.zeros(B,bool);lp=np.zeros((B,MAXR));act=np.full((B,MAXR),PAD)
    for t in range(MAXR):
        lg=m.lm(x).v[:,-1]/temp;pr=np.exp(lg-lg.max(1,keepdims=True));pr/=pr.sum(1,keepdims=True)
        a=pr.argmax(1) if greedy else (pr.cumsum(1)>r.random((B,1))).argmax(1)
        a=np.where(done,PAD,a);lp[:,t]=np.where(done,0,np.log(pr[np.arange(B),a]+1e-12));act[:,t]=a;done|=a==EOS;x=np.c_[x,a]
    return act,lp
def resp(row):
    o=[]
    for t in row:
        if t==EOS or t==PAD:break
        o.append(int(t))
    return o
def oracle(op,a,R,kind='A'):
    """真实偏好（标注员不直接告诉模型）"""
    t=truth(op,a);s=0.
    if R==t:s+=3
    else:
        c=sum(u==v for u,v in zip(R,t));s+=c/len(t)
    if BAD in R:s-=3
    s-=.3*max(0,len(R)-len(t))
    if kind=='B' and R[:1]==[HEDGE]:s+=2.5   # 喜欢先说“也许”的标注员
    if kind=='B' and R[1:]==t and R[:1]==[HEDGE]:s+=3
    return s
def evaluate(m,seed,B=300):
    r=np.random.default_rng(seed);Pr=[prompt(r) for _ in range(B)];P=np.array([p for p,_,_ in Pr]);act,_=sample(m,P,r);Rs=[resp(x) for x in act]
    ok=np.mean([R==truth(op,a) for R,(_,op,a) in zip(Rs,Pr)]);bad=np.mean([BAD in R for R in Rs]);sc=np.mean([oracle(op,a,R) for R,(_,op,a) in zip(Rs,Pr)])
    ask=np.mean([bool(R) and R[0] in(REP,REV,MAX) for R in Rs]);hed=np.mean([R[:1]==[HEDGE] for R in Rs])
    # 公开任务：下一个字母（不在人类数据里）
    Q=np.array([[NXT,A0+x,ARR] for x in range(5)]);lg=m.lm(Q).v[:,-1];pub=float(np.mean(lg.argmax(1)==A0+np.arange(1,6)));pl=lg-lg.max(1,keepdims=True);pp=np.exp(pl)/np.exp(pl).sum(1,keepdims=True);ll=float(np.mean(np.log(pp[np.arange(5),A0+np.arange(1,6)]+1e-12)))
    return dict(ok=float(ok),bad=float(bad),score=float(sc),ask=float(ask),hedge=float(hed),public=pub,public_ll=ll)
def win(m1,m2,seed,B=300):
    r=np.random.default_rng(seed);Pr=[prompt(r) for _ in range(B)];P=np.array([p for p,_,_ in Pr]);a1,_=sample(m1,P,r);a2,_=sample(m2,P,r);w=0
    for x,y,(_,op,a) in zip(a1,a2,Pr):
        s,t=oracle(op,a,resp(x)),oracle(op,a,resp(y));w+=1 if s>t else .5 if s==t else 0
    return w/B
def sft(w,dims,r,N,steps,seed):
    m=clone(w,*dims);o=Adam(m.tr);D=[]
    for _ in range(N):
        p,op,a=prompt(r);D.append(p+truth(op,a)+[EOS])
    for st in range(1,steps+1):
        idx=r.integers(0,N,32);x,y,wt=lmbatch([D[i] for i in idx]);wt=wt*(np.arange(x.shape[1])[None,:]>=4);v,p,g=ce(m.lm(x),y,wt);p.backward(g);m.P.g=None;o.step(6e-4*(1 if st<.8*steps else .3))
    return m
class RM(Enc):
    def __init__(s,seed):
        super().__init__(V,L,d=32,h=4,ff=64,N=2,seed=seed,causal=True);s.P.v=sinus(L,32);s.head=T(np.random.default_rng(seed).normal(0,.03,(32,1)));s.ps.append(s.head);s.tr=[p for p in s.ps if p is not s.P]
    def score(s,x):
        n=(x!=PAD).sum(1)-1;h=s.encode(x);B,Ln,_=h.v.shape;mask=np.zeros((B,Ln,1));mask[np.arange(B),n]=1
        return ((h*mask)@s.head),n
def trainrm(rm,gen,r,prompts,K,steps,kind='A',noise=.2):
    """每个提示 K 个输出，排序后两两比较（一批内共享前向，对应论文的省算力做法）。"""
    o=Adam(rm.tr);hist=[]
    for st in range(1,steps+1):
        P=[prompt(r) for _ in range(8)];Xs=[];pairs=[]
        arr=np.array([p for p,_,_ in P]);rows=[]
        for k in range(K):
            act,_=sample(gen[int(r.integers(0,len(gen)))],arr,r);rows.append(act)
        idx=0
        for i,(p,op,a) in enumerate(P):
            sc=[]
            for k in range(K):
                R=resp(rows[k][i]);Xs.append(p+R+[EOS]);sc.append(oracle(op,a,R,kind)+r.normal(0,noise*2))
            for u in range(K):
                for v in range(K):
                    if sc[u]>sc[v]+1e-9:pairs.append((idx+u,idx+v))
            idx+=K
        x=pad(Xs);f,_=rm.score(x);rs=f.v.sum((1,2)) if f.v.ndim==3 else f.v.ravel()
        rr=f.v.reshape(len(Xs),-1).sum(1);g=np.zeros(len(Xs));ls=0.
        for u,v in pairs:
            d=rr[u]-rr[v];q=1/(1+np.exp(d));g[u]-=q;g[v]+=q;ls+=np.log1p(np.exp(-d))
        n=max(1,len(pairs));sd=np.zeros(f.v.shape);B2=f.v.shape[0];sd2=(g/n)
        for i in range(B2):
            m_=np.zeros(f.v[i].shape);ln=(x[i]!=PAD).sum()-1;m_[ln,0]=sd2[i];sd[i]=m_
        f.backward(sd);rm.P.g=None;o.step(1e-3*(1 if st<.8*steps else .3))
        if st%max(1,steps//6)==0:hist.append((st,round(float(ls/n),3)))
    return hist
def rmreward(rm,P,act):
    Xs=[list(p)+resp(a)+[EOS] for p,a in zip(P,act)];x=pad(Xs);f,n=rm.score(x);return f.v.reshape(len(Xs),-1).sum(1)
def ppo(w,dims,rm,sftm,r,steps,beta=.02,gamma=0.,clip=.2,lr=6e-4,B=64,log=None,trueeval=True):
    m=clone(w,*dims);o=Adam(m.tr);hist=[];base=0.
    # 奖励偏移：让 SFT 平均为 0（对应论文用偏置把示范的平均分调到 0）
    for st in range(1,steps+1):
        P=np.array([prompt(r)[0] for _ in range(B)]);act,lp=sample(m,P,r);Rw=rmreward(rm,P,act)
        x=np.c_[P,act];lg=sftm.lm(x[:,:-1]).v[:,P.shape[1]-1:];pl=lg-lg.max(-1,keepdims=True);ps=np.exp(pl)/np.exp(pl).sum(-1,keepdims=True)
        lpS=np.log(ps[np.arange(B)[:,None],np.arange(MAXR)[None,:],act]+1e-12)*(act!=PAD)
        kl=(lp-lpS).sum(1);adv=Rw-beta*kl;base=.9*base+.1*adv.mean() if st>1 else adv.mean();adv2=(adv-base)/(adv.std()+1e-6)
        for inner in range(2):      # 把一批切两份，只过一遍（论文是 512 切 8 份）
            lgt=m.lm(x[:,:-1]);probs=lgt.softmax();idx0=np.arange(B)[:,None];idx1=np.arange(P.shape[1]-1,x.shape[1]-1)[None,:];pa=probs.v[idx0,idx1,act]+1e-12
            ratio=np.exp(np.log(pa)*(act!=PAD)-lp)
            cl=np.clip(ratio,1-clip,1+clip);weight=(np.where(adv2[:,None]*ratio<adv2[:,None]*cl,ratio,cl))*adv2[:,None]*(act!=PAD)   # min(未剪裁,剪裁)
            sd=np.zeros(probs.v.shape);sd[idx0,idx1,act]=-weight/pa/B;probs.backward(sd);m.P.g=None
            if gamma>0:
                xx,yy,wt=lmbatch([internet(r) for _ in range(16)]);v,p2,g2=ce(m.lm(xx),yy,wt);p2.backward(g2*gamma);m.P.g=None
            o.step(lr)
        if st%max(1,steps//10)==0:
            e=evaluate(m,700+st);hist.append(dict(st=st,rm=float(Rw.mean()),kl=float(kl.mean()),score=e['score'],ok=e['ok'],public=e['public'],public_ll=e['public_ll']))
    return m,hist
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 3000;Sp=int(sys.argv[3]) if len(sys.argv)>3 else 500
    if S<=0:
        rm=RM(0);r=np.random.default_rng(0);m=make(16,2,32,0);print(trainrm(rm,[m],r,None,2,2));
        x=np.array([[REP,A0,A0+1,A0+2,ARR,A0,A0+1,EOS]]);f,_=rm.score(x);print('score',f.v.ravel()[:3]);sys.exit()
    r=np.random.default_rng(sd+3);t0=time.time();out={};big=(32,4,64);small=(16,2,32)
    M=make(*big,sd);pretrain(M,r,S);G=get(M);out['gpt']=evaluate(M,1000+sd);print(sd,'GPT',out['gpt'],round(time.time()-t0),flush=True)
    Sm=make(*small,sd);pretrain(Sm,r,S);Gs=get(Sm);out['gpt_small']=evaluate(Sm,1000+sd);print(sd,'small',out['gpt_small'],round(time.time()-t0),flush=True)
    F=sft(G,big,r,30,150,sd);out['sft']=evaluate(F,1000+sd);print(sd,'SFT',out['sft'],round(time.time()-t0),flush=True)
    Fs=sft(Gs,small,r,30,150,sd);out['sft_small']=evaluate(Fs,1000+sd)
    rm=RM(sd);out['rm_hist']=trainrm(rm,[F,clone(G,*big)],r,None,4,500);print(sd,'RM',out['rm_hist'],round(time.time()-t0),flush=True)
    P,h=ppo(get(F),big,rm,F,r,Sp,beta=.02);out['ppo']=evaluate(P,1000+sd);out['ppo_hist']=h;print(sd,'PPO',out['ppo'],round(time.time()-t0),flush=True)
    Px,hx=ppo(get(F),big,rm,F,r,Sp,beta=.02,gamma=1.);out['ppox']=evaluate(Px,1000+sd);out['ppox_hist']=hx;print(sd,'PPOx',out['ppox'],flush=True)
    P0,h0=ppo(get(F),big,rm,F,r,Sp,beta=0.);out['ppo_nokl']=evaluate(P0,1000+sd);out['ppo_nokl_hist']=h0;print(sd,'noKL',out['ppo_nokl'],flush=True)
    Ps,hs=ppo(get(Fs),small,rm,Fs,r,Sp,beta=.02);out['ppo_small']=evaluate(Ps,1000+sd);print(sd,'small PPO',out['ppo_small'],round(time.time()-t0),flush=True)
    Gm=clone(G,*big)
    out['wins']=dict(ppox_gpt=win(Px,Gm,2000+sd),ppo_gpt=win(P,Gm,2000+sd),sft_gpt=win(F,Gm,2000+sd),smallppo_gpt=win(Ps,Gm,2000+sd),ppo_sft=win(P,F,2000+sd),gpt_sft=win(Gm,F,2000+sd),ppox_sft=win(Px,F,2000+sd),smallppo_sft=win(Ps,F,2000+sd))
    # 换一批标注员：喜欢回答前先说“也许”
    rb=RM(sd+1);out['rmB']=trainrm(rb,[F,clone(G,*big)],r,None,4,500,kind='B');Pb,_=ppo(get(F),big,rb,F,r,Sp,beta=.02);out['ppo_b']=evaluate(Pb,1000+sd);print(sd,'labeler B',out['ppo_b'],flush=True)
    out['sec']=round(time.time()-t0);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False);print('done',out['wins'],out['sec'])
    if sd==0:
        for n,mm in(('gpt',Gm),('sft',F),('ppo',P),('ppox',Px)):np.savez(f'm_{n}.npz',**{f'p{i}':p.v for i,p in enumerate(mm.ps)})
