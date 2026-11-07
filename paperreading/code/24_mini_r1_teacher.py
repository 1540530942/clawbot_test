#!/usr/bin/env python3
# 第 24 期 R1 强老师蒸馏
# 单文件版：下面依次是极小的自动微分、迷你 Transformer 和实验本身，只依赖 NumPy。
import gc
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
        if s.n%20==0: gc.collect()   # 计算图可能有引用环，定期回收防止内存涨
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

"""第 24 期玩具实验：连加连减（模 5）上的迷你 DeepSeek-R1。
预训练语料里一半直接答，一半写步骤。回答格式统一为：思考 ; 答案 <e>。
奖励只看答案对不对，外加一点格式奖励，和“思考写多长”无关。
比较：GRPO、不减基线、语料没有步骤、奖励长度（奖励黑客）、基座太弱、冷启动四阶段流水线、蒸馏和小模型直接强化学习。
这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
MOD=5;NOISE=.02
vocab=list('01234')+['+','-','=',';','<e>','<p>']
ID={c:i for i,c in enumerate(vocab)};PLUS,MINUS,EQ,SEMI,EOS,PAD=[ID[c] for c in('+','-','=',';','<e>','<p>')]
V=len(vocab);L=16;MAXR=6;KS=(2,3,4);BIG=(32,4,64);SMALL=(16,2,32)
def sinus(L,d):
    p=np.arange(L)[:,None];t=10000.0**(-2*np.arange(d//2)/d);s=np.zeros((L,d));s[:,0::2]=np.sin(p*t);s[:,1::2]=np.cos(p*t);return s
def problem(r,K):
    ds=r.integers(0,MOD,K);ops=r.integers(0,2,K-1);pr=[int(ds[0])];run=[];c=int(ds[0])
    for o,d in zip(ops,ds[1:]):
        pr+=[PLUS if o==0 else MINUS,int(d)];c=(c+int(d))%MOD if o==0 else (c-int(d))%MOD;run.append(c)
    return pr+[EQ],run,c
def corpus(r,ps):
    """预训练语料：ps 的比例写步骤，10% 是乱写，其余直接答"""
    p,run,a=problem(r,int(r.choice(KS)));u=r.random()
    if u<ps:return p+run+[SEMI,a,EOS]
    if u<ps+NOISE:
        n=int(r.integers(0,4));return p+[int(x) for x in r.integers(0,MOD,n)]+[SEMI,int(r.integers(0,MOD)),EOS]
    return p+[SEMI,a,EOS]
def pad(S):
    n=max(map(len,S));x=np.full((len(S),n),PAD)
    for i,s in enumerate(S):x[i,:len(s)]=s
    return x
def make(dims,seed):
    d,h,ff=dims;m=Enc(V,L,d=d,h=h,ff=ff,N=2,seed=seed,causal=True);m.P.v=sinus(L,d);m.tr=[p for p in m.ps if p is not m.P];m.dims=dims;return m
def get(m):return [p.v.copy() for p in m.ps]
def clone(m):
    n=make(m.dims,0)
    for p,q in zip(n.ps,m.ps):p.v=q.v.copy()
    return n
def fit(m,gen,steps,lr,B=32):
    o=Adam(m.tr)
    for st in range(1,steps+1):
        x=pad([gen() for _ in range(B)]);X,y=x[:,:-1],x[:,1:];v,p,g=ce(m.lm(X),y,(y!=PAD).astype(float));p.backward(g);m.P.g=None;o.step(min(lr,lr*st/100)*(.3 if st>.8*steps else 1))
    return m
def probs(lg):
    l=lg-lg.max(-1,keepdims=True);q=np.exp(l);return q/q.sum(-1,keepdims=True)
def sample(m,P,r,temp=1.):
    """返回 动作、是否有效、每步对数概率（按采样温度）"""
    B=len(P);x=P.copy();done=np.zeros(B,bool);lp=np.zeros((B,MAXR));act=np.full((B,MAXR),PAD);val=np.zeros((B,MAXR),bool)
    for t in range(MAXR):
        q=probs(m.lm(x).v[:,-1]/temp);a=(q.cumsum(1)>r.random((B,1))).argmax(1);a=np.where(done,PAD,a)
        val[:,t]=~done;lp[:,t]=np.where(done,0,np.log(q[np.arange(B),a]+1e-12));act[:,t]=a;done=done|(a==EOS)|(a==PAD);x=np.c_[x,a]
    return act,val,lp
def parse(row):
    """(思考长度, 答案或 -1, 思考内容)。格式：若干数字 ; 一个数字 <e>"""
    o=[]
    for t in row:
        if t==EOS or t==PAD:break
        o.append(int(t))
    if row[len(o)]!=EOS if len(o)<len(row) else True:return len(o),-1,o
    if len(o)<2 or o[-2]!=SEMI or o[-1]>=MOD or any(t>=MOD for t in o[:-2]):return len(o),-1,o
    return len(o)-2,o[-1],o[:-2]
def rule_reward(row,ans,kind='acc'):
    n,a,_=parse(row);fmt=.1 if a>=0 else 0.
    if kind=='acc':return (1. if a==ans else 0.)+fmt
    if kind=='len':return n/ (MAXR-2)+fmt    # 代理奖励：越长越好，代替学坏了的奖励模型
def seqlogp(m,x,act,val,plen):
    lgt=m.lm(x[:,:-1]);pr=lgt.softmax();B=len(x);i0=np.arange(B)[:,None];i1=np.arange(plen-1,x.shape[1]-1)[None,:];pa=pr.v[i0,i1,act]+1e-12
    return pr,pa,i0,i1,(np.log(pa)*val).sum(1)
def grpo_step(m,o,ref,r,Q,G,lr,eps=.2,beta=0.,base='group',kind='acc',inner=1,K=None):
    K=K or int(r.choice(KS));Pr=[problem(r,K) for _ in range(Q)];P=np.repeat(np.array([p for p,_,_ in Pr]),G,0);A=np.repeat([a for _,_,a in Pr],G)
    act,val,lpold=sample(m,P,r);old=lpold.sum(1);R=np.array([rule_reward(a,y,kind) for a,y in zip(act,A)]);plen=P.shape[1]
    if base=='group':
        Rg=R.reshape(Q,G);Adv=((Rg-Rg.mean(1,keepdims=True))/(Rg.std(1,keepdims=True)+1e-6)).ravel()
    elif base=='none':Adv=R.copy()
    x=np.c_[P,act];_,paR,_,_,lref=seqlogp(ref,x,act,val,plen) if ref is not None else (0,0,0,0,old)
    for _ in range(inner):
        pr,pa,i0,i1,lp=seqlogp(m,x,act,val,plen);ratio=np.exp(np.clip(lp-old,-5,5));clipr=np.clip(ratio,1-eps,1+eps)
        c=np.where(ratio*Adv<=clipr*Adv,ratio*Adv,0.)     # dJ/dlogπ：未剪裁那一支才有梯度
        rho=np.exp(np.clip(lref-lp,-3,3));c=c-beta*(1-rho)   # KL 项：d(ρ−logρ−1)/dlogπ = 1−ρ
        sd=np.zeros(pr.v.shape);sd[i0,i1,act]=-(c[:,None]*val)/pa/(Q*G);pr.backward(sd);m.P.g=None;o.step(lr)
    return float((R>=1).mean()) if kind=='acc' else float(R.mean())
def rl(m,ref,steps,seed,Q=8,G=8,lr=1e-3,log=10,tag='',**kw):
    r=np.random.default_rng(seed);o=Adam(m.tr);h=[]
    for st in range(1,steps+1):
        grpo_step(m,o,ref,r,Q,G,lr,**kw)
        if st%max(1,steps//log)==0:
            e=score(m,seed+st,B=30,cons=False);h.append(dict(st=st,**{k:e[k] for k in('pass1','think','full','empty')}))
            if st%max(1,steps//4)==0:print(tag,st,e['pass1'],e['think'],e['full'],e['empty'],flush=True)
    return h
def score(m,seed,B=100,T=.6,cons=True):
    r=np.random.default_rng(seed);res={};tot=[];thk=[];full=[];emp=[];cor={'full':[],'empty':[],'other':[]};vt=[]
    for K in KS:
        Pr=[problem(r,K) for _ in range(B)];act,_,_=sample(m,np.array([p for p,_,_ in Pr]),r,T);ok=[]
        for row,(p,run,a) in zip(act,Pr):
            n,y,th=parse(row);c=y==a;ok.append(c);thk.append(n if y>=0 else 0);f=(th==run);full.append(f);e=(n==0 and y>=0);emp.append(e);cor['full' if f else 'empty' if e else 'other'].append(c)
        res[K]=float(np.mean(ok));tot+=ok
        if cons:
            Pv=[problem(r,K) for _ in range(30)];P=np.repeat(np.array([p for p,_,_ in Pv]),16,0);act,_,_=sample(m,P,r,T);ys=np.array([parse(a)[1] for a in act]).reshape(30,16)
            for i,(p,_,a) in enumerate(Pv):
                u=ys[i][ys[i]>=0];vt.append(len(u)>0 and np.bincount(u,minlength=MOD).argmax()==a)
    out=dict(K=res,pass1=float(np.mean(tot)),think=float(np.mean(thk)),full=float(np.mean(full)),empty=float(np.mean(emp)),acc_full=float(np.mean(cor['full'])) if cor['full'] else None,acc_empty=float(np.mean(cor['empty'])) if cor['empty'] else None)
    if cons:out['cons16']=float(np.mean(vt))
    return out
def rejection(m,r,n,kept=False):
    """采样，只留对的（还要求格式好、写了步骤时写对），类似论文 §2.3.3"""
    D=[]
    while len(D)<n:
        K=int(r.choice(KS));Pr=[problem(r,K) for _ in range(32)];act,_,_=sample(m,np.array([p for p,_,_ in Pr]),r,.6)
        for row,(p,run,a) in zip(act,Pr):
            k,y,th=parse(row)
            if y==a:D.append(p+th+[SEMI,a,EOS])
    return D[:n]
def main():
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 5000;Sr=int(sys.argv[3]) if len(sys.argv)>3 else 200
    if S<=0:
        # 数值核对 GRPO 梯度：比较手写系数和 J 的有限差分
        m=make((16,2,32),0);ref=clone(m);ref.E.v=ref.E.v+.05;r=np.random.default_rng(1);K=3;Pr=[problem(r,K) for _ in range(2)];P=np.repeat(np.array([p for p,_,_ in Pr]),4,0);act,val,lo=sample(m,P,r);old=lo.sum(1);Adv=r.normal(0,1,8);x=np.c_[P,act];plen=P.shape[1];beta=.3;eps=.2
        _,_,_,_,lr_=seqlogp(ref,x,act,val,plen);m.E.v[3,2]+=.03
        def J():
            _,_,_,_,lp=seqlogp(m,x,act,val,plen);ra=np.exp(lp-old);rho=np.exp(lr_-lp);return float((np.minimum(ra*Adv,np.clip(ra,1-eps,1+eps)*Adv)-beta*(rho-np.log(rho)-1)).mean())
        for p in m.ps:p.g=None
        pr,pa,i0,i1,lp=seqlogp(m,x,act,val,plen);ra=np.exp(lp-old);cl=np.clip(ra,1-eps,1+eps);c=np.where(ra*Adv<=cl*Adv,ra*Adv,0.);rho=np.exp(lr_-lp);c=c-beta*(1-rho);sd_=np.zeros(pr.v.shape);sd_[i0,i1,act]=-(c[:,None]*val)/pa/8;pr.backward(sd_);g=m.E.g[3,2];e=1e-6;m.E.v[3,2]+=e;a=J();m.E.v[3,2]-=2*e;b=J();m.E.v[3,2]+=e;print('手写',-g,'数值',(a-b)/2/e);sys.exit()
    t0=time.time();r=np.random.default_rng(sd+5);out={}
    def base(dims,ps,tag,steps=None):
        steps=steps or S
        m=make(dims,sd);fit(m,lambda:corpus(r,ps),steps,2e-3);out[tag]=score(m,3000+sd);print(sd,tag,out[tag],round(time.time()-t0),flush=True);return m
    Ps=.5
    B=base(BIG,Ps,'base');Bd=base(BIG,0.,'base_direct');Bs=base(SMALL,Ps,'base_small')
    Bw=base(BIG,Ps,'base_weak',S*3//10)
    def arm(tag,start,ref=None,steps=Sr,**kw):
        m=clone(start);h=rl(m,ref if ref is not None else start,steps,sd+7,tag=f'{sd} {tag}',**kw);out[tag]=score(m,3000+sd);out[tag]['hist']=h;print(sd,tag,{k:v for k,v in out[tag].items() if k!='hist'},round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False);return m
    # 第一轮蒸馏的老师（流水线 R1）正确率只有约四成。这里换成 GRPO 强基座（约 98%），只比较老师的影响
    Z=arm('zero',B);Dz=rejection(Z,r,2000);Ds=fit(clone(Bs),lambda:Dz[int(r.integers(0,2000))],900,1e-3);out['distill_zero']=score(Ds,3000+sd);print(sd,'distill_zero',out['distill_zero'],flush=True)
    out['sec']=round(time.time()-t0);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False);print('done',out['sec'])
    if sd==0:
        for n,mm in():np.savez(f'm_{n}.npz',**{f'p{i}':p.v for i,p in enumerate(mm.ps)})
if __name__=='__main__':
    main()
