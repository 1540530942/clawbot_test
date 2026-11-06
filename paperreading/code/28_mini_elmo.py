#!/usr/bin/env python3
# 迷你 ELMo：双向 LSTM 语言模型，冻结后用线性分类器判断词义和词性
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

"""第 28 期玩具实验：迷你 ELMo。
合成语料里有多义词（bank 河岸/银行，bat 蝙蝠/球棒，play 戏剧/玩耍，fly 飞/苍蝇，light 灯/轻的）。
先无监督训练双向语言模型（2 层 LSTM、共享词向量和 softmax、层间残差），然后冻结，
用线性分类器在各层表示上预测词义和词性，比较词向量、第 1 层、第 2 层、平均、可学习权重。这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
# 模板：(词, 词性, 词义)；词义只给多义词
T_=[
 ('the man went to the bank to get money',{'bank':'bank:金融'}),
 ('the woman works at the bank in the town',{'bank':'bank:金融'}),
 ('she put her money in the bank last week',{'bank':'bank:金融'}),
 ('the bank gave the man some money',{'bank':'bank:金融'}),
 ('the fish swam near the bank of the river',{'bank':'bank:河岸'}),
 ('the boy sat on the bank and saw the water',{'bank':'bank:河岸'}),
 ('grass grew along the bank of the lake',{'bank':'bank:河岸'}),
 ('the bank was green near the river',{'bank':'bank:河岸'}),
 ('the boy hit the ball with the bat in the game',{'bat':'bat:球棒'}),
 ('he held the bat and ran after the ball',{'bat':'bat:球棒'}),
 ('the team gave her a new bat for the game',{'bat':'bat:球棒'}),
 ('the bat flew out of the cave at night',{'bat':'bat:蝙蝠'}),
 ('a small bat ate insects in the dark cave',{'bat':'bat:蝙蝠'}),
 ('at night the bat hunted in the dark',{'bat':'bat:蝙蝠'}),
 ('the actors were in a good play at the theater',{'play':'play:戏剧'}),
 ('we saw the play at the theater last week',{'play':'play:戏剧'}),
 ('the play had a sad end and good actors',{'play':'play:戏剧'}),
 ('the children play with the ball in the park',{'play':'play:玩耍'}),
 ('they play in the park after the game',{'play':'play:玩耍'}),
 ('the boys play a game with the team',{'play':'play:玩耍'}),
 ('birds fly over the river at night',{'fly':'fly:飞'}),
 ('the planes fly over the town in the morning',{'fly':'fly:飞'}),
 ('they fly to the town in the morning',{'fly':'fly:飞'}),
 ('a fly sat on the food in the room',{'fly':'fly:苍蝇'}),
 ('the fly ate the sweet food on the table',{'fly':'fly:苍蝇'}),
 ('she saw a small fly on the table',{'fly':'fly:苍蝇'}),
 ('she turned on the light in the room',{'light':'light:灯'}),
 ('the light in the room was bright at night',{'light':'light:灯'}),
 ('he saw the bright light on the table',{'light':'light:灯'}),
 ('the bag was light and easy to carry',{'light':'light:轻'}),
 ('she can carry the light bag to the town',{'light':'light:轻'}),
 ('the small box was light and she can carry it',{'light':'light:轻'}),
]
POS={'the':'冠','a':'冠','her':'冠','some':'冠','his':'冠','and':'连','man':'名','woman':'名','boy':'名','fish':'名','money':'名','town':'名','river':'名','water':'名','lake':'名','grass':'名','ball':'名','game':'名','team':'名','cave':'名','night':'名','insects':'名','dark':'名','actors':'名','theater':'名','end':'名','children':'名','park':'名','boys':'名','birds':'名','planes':'名','morning':'名','food':'名','room':'名','table':'名','bag':'名','box':'名','week':'名','he':'代','she':'代','they':'代','we':'代','it':'代','went':'动','get':'动','works':'动','put':'动','gave':'动','swam':'动','sat':'动','saw':'动','grew':'动','was':'动','hit':'动','held':'动','ran':'动','flew':'动','ate':'动','hunted':'动','were':'动','had':'动','turned':'动','carry':'动','can':'动','to':'介','at':'介','in':'介','near':'介','of':'介','on':'介','along':'介','with':'介','after':'介','for':'介','out':'介','over':'介','last':'形','new':'形','small':'形','good':'形','sad':'形','sweet':'形','bright':'形','green':'形','easy':'形'}
SENSEPOS={'bank:金融':'名','bank:河岸':'名','bat:球棒':'名','bat:蝙蝠':'名','play:戏剧':'名','play:玩耍':'动','fly:飞':'动','fly:苍蝇':'名','light:灯':'名','light:轻':'形'}
FILL={'man':['man','woman','boy'],'woman':['man','woman','boy'],'boy':['man','woman','boy'],'small':['small','new','good'],'good':['small','new','good'],'town':['town','park'],'park':['town','park'],'river':['river','lake'],'lake':['river','lake'],'room':['room','cave'] ,'night':['night','morning'],'morning':['night','morning'],'she':['she','he'],'he':['she','he']}
vocab=['<p>','<s>','</s>']+sorted({w for s,_ in T_ for ws in s.split() for w in FILL.get(ws,[ws])})
ID={w:i for i,w in enumerate(vocab)};PAD=0;BOS=1;EOSS=2;V=len(vocab)
SENSES=sorted(SENSEPOS);TAGS=sorted(set(POS.values())|set(SENSEPOS.values()));
def sample(r):
    s,sm=T_[int(r.integers(0,len(T_)))];ws=[]
    for w in s.split():
        c=FILL.get(w)
        if c and not(w=='room'):w=c[int(r.integers(0,len(c)))]
        ws.append(w)
    tags=[];sense=[]
    for w in ws:
        if w in sm:sense.append(SENSES.index(sm[w]));tags.append(TAGS.index(SENSEPOS[sm[w]]))
        else:sense.append(-1);tags.append(TAGS.index(POS[w]))
    return ws,tags,sense
sg=lambda z:1/(1+np.exp(-np.clip(z,-30,30)))
def sigm(x):
    p=sg(x.v);r=T(p,(x,));r.bk=lambda g:(g*p*(1-p),);return r
def tanh(x):
    p=np.tanh(x.v);r=T(p,(x,));r.bk=lambda g:(g*(1-p*p),);return r
def cat(a,b):
    r=T(np.concatenate([a.v,b.v],-1),(a,b));n=a.v.shape[-1];r.bk=lambda g:(g[...,:n],g[...,n:]);return r
def part(a,i,j):
    r=T(a.v[...,i:j],(a,))
    def bk(g):
        o=np.zeros_like(a.v);o[...,i:j]=g;return (o,)
    r.bk=bk;return r
def pick(E,ids):
    r=T(E.v[ids],(E,))
    def bk(g):
        o=np.zeros_like(E.v);np.add.at(o,ids,g);return (o,)
    r.bk=bk;return r
class BiLM:
    def __init__(s,n,seed):
        r=np.random.default_rng(seed);s.n=n;s.ps=[];s.nm=[]
        def G(a,b,sd,name):
            t=T(r.normal(0,sd,(a,b)));s.ps.append(t);s.nm.append(name);return t
        s.E=G(V,n,.1,'E');s.Wo=G(n,V,.1,'Wo');s.cell={}
        for d in 'fb':
            for l in (0,1):
                s.cell[d,l]=dict(W=G(2*n,4*n,.1,f'{d}{l}W'))
                t=np.zeros(4*n);t[:n]=1.;b=T(t);s.ps.append(b);s.nm.append(f'{d}{l}b');s.cell[d,l]['b']=b
    def lstm(s,c,x,h,m):
        a=cat(x,h)@c['W']+T(c['b'].v*0)+c['b'];n=s.n;f=sigm(part(a,0,n));i=sigm(part(a,n,2*n));o=sigm(part(a,2*n,3*n));g=tanh(part(a,3*n,4*n));M=f*m+i*g;return o*tanh(M),M
    def direction(s,d,ids):
        B,Tn=ids.shape;h=[T(np.zeros((B,s.n))) for _ in range(2)];m=[T(np.zeros((B,s.n))) for _ in range(2)];H0=[];H1=[]
        for t in range(Tn):
            x=pick(s.E,ids[:,t]);h[0],m[0]=s.lstm(s.cell[d,0],x,h[0],m[0]);h[1],m[1]=s.lstm(s.cell[d,1],h[0],h[1],m[1]);H0.append(h[0]);H1.append(h[0]+h[1])   # 层间残差
        return H0,H1
    def loss(s,X):
        """X:(B,T) 含 <s> ... </s>。返回 损失值、反传闭包"""
        B,Tn=X.shape;Ls=[];tot=0.;cnt=0
        F0,F1=s.direction('f',X[:,:-1]);R0,R1=s.direction('b',X[:,:0:-1])
        def add(Hs,Y):
            nonlocal tot,cnt
            for t,h in enumerate(Hs):
                y=Y[:,t];w=(y!=PAD)*1.;p=(h@s.Wo).softmax();py=p.v[np.arange(B),y]+1e-12;tot-=(w*np.log(py)).sum();cnt+=w.sum();sd=np.zeros(p.v.shape);sd[np.arange(B),y]=-w/py;Ls.append((p,sd))
        add(F1,X[:,1:]);add(R1,X[:,:0:-1][:,1:] if False else X[:,:-1][:,::-1]);
        return tot/cnt,Ls,cnt
    def reps(s,ids):
        """返回 x, h1, h2，形状 (B,T,·)；去掉首尾"""
        X=np.array(ids);F0,F1=s.direction('f',X);R0,R1=s.direction('b',X[:,::-1]);R0=R0[::-1];R1=R1[::-1]
        x=s.E.v[X];x=np.concatenate([x,x],-1);h1=np.stack([np.concatenate([a.v,b.v],-1) for a,b in zip(F0,R0)],1);h2=np.stack([np.concatenate([a.v,b.v],-1) for a,b in zip(F1,R1)],1);return x,h1,h2
def lmbatch(r,B):
    S=[[BOS]+[ID[w] for w in sample(r)[0]]+[EOSS] for _ in range(B)];n=max(map(len,S));X=np.zeros((B,n),int)
    for i,s_ in enumerate(S):X[i,:len(s_)]=s_
    return X
def train_lm(m,steps,seed):
    r=np.random.default_rng(seed+9);o=Adam(m.ps);h=[]
    for st in range(1,steps+1):
        X=lmbatch(r,16);L,Ls,c=m.loss(X);seeds=[g/c for _,g in Ls];root=T(0.,tuple(p for p,_ in Ls));root.bk=lambda g:tuple(seeds);root.backward()
        o.step(3e-3*(1 if st<.8*steps else .3))
        if st%max(1,steps//10)==0:h.append((st,round(float(np.exp(L)),2)))
    return h
def dataset(m,n,seed):
    """抽 n 句，算三层表示。返回 特征 (N,3,d)、词义、词性、词 id"""
    r=np.random.default_rng(seed);F=[];Se=[];Ta=[];Wd=[];Sent=[]
    for i in range(n):
        ws,tags,sense=sample(r);ids=[[BOS]+[ID[w] for w in ws]+[EOSS]];x,h1,h2=m.reps(ids)
        for t in range(len(ws)):
            F.append(np.stack([x[0,t+1],h1[0,t+1],h2[0,t+1]]));Se.append(sense[t]);Ta.append(tags[t]);Wd.append(ID[ws[t]]);Sent.append(i)
    return np.array(F),np.array(Se),np.array(Ta),np.array(Wd)
def clf(F,y,Ft,yt,k,mode,steps=300,lam=0.,seed=0):
    """线性分类。mode: 0/1/2 只用该层；avg 平均；learn 学习 s、γ（式 1）"""
    r=np.random.default_rng(seed);d=F.shape[2];W=np.zeros((d,k));b=np.zeros(k);s_=np.zeros(3);gam=1.;mw=[0,0,0,0];lr=.05;hist=None
    mu=F.mean(0,keepdims=True);sd=F.std(0,keepdims=True)+1e-6;F=(F-mu)/sd;Ft=(Ft-mu)/sd
    def mix(A):
        if mode in(0,1,2):return A[:,mode]
        if mode=='avg':return A.mean(1)
        p=np.exp(s_)/np.exp(s_).sum();return (A*p[None,:,None]).sum(1)
    M=[np.zeros_like(W),np.zeros_like(b),np.zeros(3)];U=[np.zeros_like(W),np.zeros_like(b),np.zeros(3)]
    for st in range(1,steps+1):
        X=mix(F);lg=X@W+b;lg-=lg.max(1,keepdims=True);q=np.exp(lg);q/=q.sum(1,keepdims=True);e=q.copy();e[np.arange(len(y)),y]-=1;gW=X.T@e/len(y)+1e-3*W;gb=e.mean(0)
        G=[gW,gb,np.zeros(3)]
        if mode=='learn':
            p=np.exp(s_)/np.exp(s_).sum();dL=np.einsum('ni,nji->j',e@W.T,F)/len(y);G[2]=p*(dL-(p*dL).sum())+lam*s_
        for i,(a,g) in enumerate(zip((W,b,s_),G)):
            M[i]=.9*M[i]+.1*g;U[i]=.99*U[i]+.01*g*g;a-=lr*(M[i]/(1-.9**st))/(np.sqrt(U[i]/(1-.99**st))+1e-8)
    Xt=mix(Ft);acc=float(((Xt@W+b).argmax(1)==yt).mean());return acc,(np.exp(s_)/np.exp(s_).sum()).tolist(),W,b,mu,sd
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 2000;n=int(sys.argv[3]) if len(sys.argv)>3 else 48;out={};t0=time.time()
    if S<=0:
        m=BiLM(6,0);r=np.random.default_rng(0);X=lmbatch(r,2);L,Ls,c=m.loss(X);seeds=[g/c for _,g in Ls];root=T(0.,tuple(p for p,_ in Ls));root.bk=lambda g:tuple(seeds);root.backward();bad=0
        for nm,p in zip(m.nm,m.ps):
            g=p.g.copy();i=np.unravel_index(int(np.argmax(abs(g))),g.shape);e=1e-5;p.v[i]+=e;a=m.loss(X)[0];p.v[i]-=2*e;b=m.loss(X)[0];p.v[i]+=e;bad=max(bad,abs((a-b)/2/e-g[i])/(abs(g[i])+1e-9))
        print('最大相对误差',bad);sys.exit()
    m=BiLM(n,sd);h=train_lm(m,S,sd);out['lm']=h;print(sd,'LM',h,round(time.time()-t0),flush=True)
    Ft,Set,Tat,Wt=dataset(m,600,50+sd);res={}
    for N in (10,30,100,300,1000):
        F,Se,Ta,W=dataset(m,N,100+sd);tr=Se>=0;tt=Set>=0;row={}
        for mode,k in ((0,'x'),(1,'h1'),(2,'h2'),('avg','avg'),('learn','learn')):
            a,sw,*_=clf(F[tr],Se[tr],Ft[tt],Set[tt],len(SENSES),mode);b,sp,*_=clf(F,Ta,Ft,Tat,len(TAGS),mode);row[k]=dict(wsd=a,pos=b,sw=sw,sp=sp)
        res[N]=row;print(sd,N,{k:(round(v['wsd'],3),round(v['pos'],3)) for k,v in row.items()},row['learn']['sw'],round(time.time()-t0),flush=True);out['res']=res;json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
    # 1-近邻词义（论文的做法）：每个词义取平均表示
    F,Se,Ta,W=dataset(m,1000,100+sd);nn={}
    for j,k in enumerate(('x','h1','h2')):
        P=np.stack([F[Se==s_,j].mean(0) for s_ in range(len(SENSES))]);tt=Set>=0;P=P/np.linalg.norm(P,axis=1,keepdims=True);Q=Ft[tt,j]/np.linalg.norm(Ft[tt,j],axis=1,keepdims=True);same=np.array([[SENSES[s_].split(':')[0]==vocab[w] for s_ in range(len(SENSES))] for w in Wt[tt]]);sc=Q@P.T;sc[~same]=-9;nn[k]=float((sc.argmax(1)==Set[tt]).mean())
    out['nn']=nn;print(sd,'1NN',nn,round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
    if sd==0:
        tr=Se>=0;a,sw,Ww,bb,mu,sdv=clf(F[tr],Se[tr],Ft[Set>=0],Set[Set>=0],len(SENSES),2);np.savez('m.npz',**{k:p.v for k,p in zip(m.nm,m.ps)},cW=Ww,cb=bb,mu=mu[0,2],sd=sdv[0,2]);print('保存',a)
