#!/usr/bin/env python3
# 第 33 期迷你 Toolformer（第 5 次运行；第 4 次 33d 结果逐项相同）
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

"""第 33 期玩具实验：迷你 Toolformer。两位数加减：加法小模型能学会，减法学得慢，看过滤能不能只在需要的地方留下调用。
（前两次：33 两位数、33b 三位数，模型都读不懂 [calc…] 前缀，过滤后几乎不剩。证据见 paperreading_evidence/33、33b。）
走完采样候选、执行、用损失过滤、在带调用的文本上微调、边生成边调用工具的全过程。
小模型没有少样本能力，所以候选由随机模板给出，这是和论文最大的差别。这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
vocab=list('0123456789')+['+','-','[',']','->','calc','cats','dogs','total','diff','note','have','and','.','<p>']
ID={c:i for i,c in enumerate(vocab)};PAD=ID['<p>'];LB,RB,AR,CALC=ID['['],ID[']'],ID['->'],ID['calc'];V=len(vocab);L=44
def num(n):return [ID[c] for c in str(n)]
def text(r,kind=None):
    a,b=int(r.integers(10,100)),int(r.integers(10,100));a,b=max(a,b),min(a,b)
    u=r.random() if kind is None else kind
    s=num(a) ,num(b)
    head=[ID['cats']]+num(a)+[ID['dogs']]+num(b)
    if u<.3:ans=a+b;return head+[ID['total']]+num(ans)+[ID['.']],'sum',(a,b)
    if u<.5:ans=a-b;return head+[ID['diff']]+num(ans)+[ID['.']],'diff',(a,b)
    if u<.8:   # 前面写过答案，模型由此学会抄数字（对应 GPT-J 的上下文能力）
        return note(r,a,b)
    return [ID['have']]+num(a)+[ID['cats'],ID['and']]+num(b)+[ID['dogs'],ID['.']],'plain',(a,b)
def note(r,a,b):
    """笔记句：句首随便怎么写，都把最后写下的数抄到答案里。GPT-J 见过海量这类文本，小模型只能这样补。
    开头的形状随机，但绝不出现 “[ calc … -> 结果 ]” 这种调用的完整格式。"""
    head=[ID['cats']]+num(a)+[ID['dogs']]+num(b);n=int(r.integers(0,200));eq=r.random()<.6
    if eq:
        p,q=int(r.integers(10,100)),int(r.integers(10,100));p,q=max(p,q),min(p,q);o='+' if r.random()<.5 else '-';n=ex(p,o,q);body=num(p)+[ID[o]]+num(q)+[AR]+num(n)
    else:body=num(n)
    op=[[ID['note']],[LB],[CALC],[LB,ID['note']],[LB,CALC],[ID['have']],[]][int(r.integers(0,7))];cl=[[],[RB],[ID['.']]][int(r.integers(0,3))]
    if eq and op==[LB,CALC] and cl==[RB]:cl=[]
    return op+body+cl+head+[ID['total' if r.random()<.5 else 'diff']]+num(n)+[ID['.']],'copy',(a,b)
def pad(S):
    n=max(map(len,S));x=np.full((len(S),n),PAD)
    for i,s in enumerate(S):x[i,:len(s)]=s
    return x
def make(seed):
    m=Enc(V,L,d=48,h=4,ff=96,N=2,seed=seed,causal=True);return m
def fit(m,gen,steps,lr):
    o=Adam(m.ps);h=[]
    for st in range(1,steps+1):
        x=pad([gen() for _ in range(32)])[:,:L];X,y=x[:,:-1],x[:,1:];v,p,g=ce(m.lm(X),y,(y!=PAD)*1.);p.backward(g);o.step(min(lr,lr*st/100)*(.3 if st>.8*steps else 1))
        if st%max(1,steps//8)==0:h.append((st,round(float(v),3)))
    return h
def probs(m,x):
    lg=m.lm(x).v;lg=lg-lg.max(-1,keepdims=True);p=np.exp(lg);return p/p.sum(-1,keepdims=True)
W=np.maximum(0,1-.2*np.arange(40));W=W/W.sum()
def Li(m,prefix,x,i):
    """前缀为 prefix 时，x 从第 i 个词起的加权交叉熵（论文的 L_i）"""
    z=prefix+x;off=len(prefix);p=probs(m,np.array([z[:-1]]))[0];s=0.
    for j in range(i,len(x)):
        w=W[j-i]
        if w>0 and off+j>=1:s-=w*np.log(p[off+j-1,x[j]]+1e-12)
    return s
def ex(a,o,b):
    return a+b if o=='+' else a-b
def ecall(a,o,b,r=None):
    t=[LB,CALC]+num(a)+[ID[o]]+num(b)
    return t+[AR]+(num(r) if r is not None else [])+[RB]
def annotate(m,r,N,tauf=1.,k=3,c=4):
    """采样位置和候选 → 执行 → 过滤。返回 C*、统计"""
    Cs=[];Cu=[];st=dict(texts=0,cand=0,kept=0,kept_ok=0);allv=[]
    while len(Cs)<N and st['texts']<N*8:
        x,ty,(a,b)=text(r);st['texts']+=1;nums=(a,b);pos=[i for i in range(1,len(x)-1)];pos=[pos[int(j)] for j in r.choice(len(pos),min(k,len(pos)),replace=False)]
        # 一定把答案开始的位置放进去一半的时候，相当于论文按 p(<API>) 选位置
        if ty in('sum','diff') and r.random()<.7:
            kw=ID['total'] if ty=='sum' else ID['diff'];pos[0]=x.index(kw)+1
        best=None
        for i in pos:
            for _ in range(c):
                p,q=(a,b) if r.random()<.7 else (b,a)
                if r.random()<.2:p=int(r.integers(10,100))
                o='+' if r.random()<.5 else '-';res=ex(p,o,q)
                if res<0:continue
                st['cand']+=1;Lp=Li(m,ecall(p,o,q,res),x,i);Lm=min(Li(m,[],x,i),Li(m,ecall(p,o,q),x,i));d=Lm-Lp
                good=(ty=='sum' and x[i]==num(a+b)[0] and res==a+b and x[i-1]==ID['total']) or (ty=='diff' and x[i-1]==ID['diff'] and res==a-b)
                allv.append((float(d),bool(good)))
                if d>=tauf and (best is None or d>best[0]):best=(d,i,p,o,q,res,good)
        if best:
            d,i,p,o,q,res,good=best;Cs.append(x[:i]+ecall(p,o,q,res)+x[i:]);st['kept']+=1;st['kept_ok']+=int(good)
        else:Cu.append(x)   # 论文：除了插入的调用，C* 和原语料内容相同，没有调用的句子也留着
    return Cs+Cu,st,allv
def gen(m,prompt,maxn=16,k=1,tool=True,fake=None):
    """贪心生成；<API> 在前 k 名就调用；看到 -> 就执行计算器。至多一次调用"""
    x=list(prompt);called=None;i=0;n=0
    while n<maxn:
        p=probs(m,np.array([x]))[0,-1];top=np.argsort(-p)
        if not tool or called:p=p.copy();p[LB]=0;top=np.argsort(-p)
        if tool and not called and LB in top[:k]:t=LB
        else:t=int(top[0])
        x.append(t);n+=1
        if t==LB:
            called=[];
            while n<maxn+10:
                t=int(np.argmax(probs(m,np.array([x]))[0,-1]));x.append(t);n+=1
                if t==AR or t==ID['.']:break
                called.append(t)
            if t!=AR:break
            s=''.join(vocab[c] for c in called if c not in(CALC,))
            try:
                import re
                g=re.fullmatch(r'(\d+)([+-])(\d+)',s);res=ex(int(g.group(1)),g.group(2),int(g.group(3)))
                if res<0:res=0
            except Exception:res=0
            x+=num(res)+[RB];n+=1;continue
        if t==ID['.']:break
    return x[len(prompt):],called
def decode_ans(out):
    o=[];d=[]; i=0;inb=False
    for t in out:
        if t==LB:inb=True;continue
        if t==RB:inb=False;continue
        if inb:continue
        if t<=9:d.append(vocab[t])
        else:break
    return int(''.join(d)) if d else -1
def evaluate(m,seed,k,tool,n=150):
    r=np.random.default_rng(seed);res={}
    for ty,kw,f in (('sum','total',lambda a,b:a+b),('diff','diff',lambda a,b:a-b)):
        ok=c=right=0
        for _ in range(n):
            a,b=int(r.integers(10,100)),int(r.integers(10,100));a,b=max(a,b),min(a,b);out,called=gen(m,[ID['cats']]+num(a)+[ID['dogs']]+num(b)+[ID[kw]],k=k,tool=tool);ans=decode_ans(out);ok+=ans==f(a,b);c+=called is not None
        res[ty]=ok/n;res[ty+'_call']=c/n
    c=0
    for _ in range(n):
        a,b=int(r.integers(10,100)),int(r.integers(10,100));out,called=gen(m,[ID['have']]+num(a)+[ID['cats'],ID['and']]+num(b)+[ID['dogs']],k=k,tool=tool);c+=called is not None
    res['plain_call']=c/n;return res
def can_read(m,seed,n=150):
    """诊断：把正确的调用和结果放在文本开头，预训练模型能照着答对吗（过滤能否生效全靠这个）"""
    r=np.random.default_rng(seed);res={}
    for ty,kw,o in (('sum','total','+'),('diff','diff','-')):
        ok=0
        for _ in range(n):
            a,b=int(r.integers(10,100)),int(r.integers(10,100));a,b=max(a,b),min(a,b);v=ex(a,o,b);p=ecall(a,o,b,v)+[ID['cats']]+num(a)+[ID['dogs']]+num(b)+[ID[kw]];out,_=gen(m,p,k=1,tool=False);ok+=decode_ans(out)==v
        res[ty]=ok/n
    return res
def can_copy(m,seed,n=150):
    r=np.random.default_rng(seed);ok=0
    for _ in range(n):
        a,b=int(r.integers(10,100)),int(r.integers(10,100));v=int(r.integers(0,200));p=[ID['note']]+num(v)+[ID['cats']]+num(a)+[ID['dogs']]+num(b)+[ID['total']];out,_=gen(m,p,k=1,tool=False);ok+=decode_ans(out)==v
    return ok/n
def call_stats(m,seed,n=150):
    """k=10 强制调用时，模型写的算式：运算符对不对、两个数对不对、最后答案有没有抄结果"""
    r=np.random.default_rng(seed);res={}
    for ty,kw,o in (('sum','total','+'),('diff','diff','-')):
        c=dict(calls=0,op=0,args=0,both=0,copied=0)
        for _ in range(n):
            a,b=int(r.integers(10,100)),int(r.integers(10,100));a,b=max(a,b),min(a,b);out,called=gen(m,[ID['cats']]+num(a)+[ID['dogs']]+num(b)+[ID[kw]],k=10)
            if called is None:continue
            c['calls']+=1;s=''.join(vocab[t] for t in called if t!=CALC);ok_o=o in s;ok_a=s.replace('+','-')==f'{a}-{b}';c['op']+=ok_o;c['args']+=ok_a;c['both']+=ok_o and ok_a
            ct=[t for t in out];
            if RB in ct:
                i=ct.index(AR);j=ct.index(RB);c['copied']+=[t for t in ct[i+1:j]]==[t for t in ct[j+1:j+1+(j-i-1)]]
        res[ty]=c
    return res
def nll(m,seed,n=200):
    r=np.random.default_rng(seed);x=pad([text(r)[0] for _ in range(n)]);return float(ce(m.lm(x[:,:-1]),x[:,1:],(x[:,1:]!=PAD)*1.)[0])
def clone(m):
    n=make(0)
    for p,q in zip(n.ps,m.ps):p.v=q.v.copy()
    return n
if __name__=='__main__':
    if len(sys.argv)>2 and int(sys.argv[2])<=0:
        r=np.random.default_rng(0);M=make(0);h=fit(M,lambda:text(r)[0],3,2e-3);print(can_read(M,1,2),annotate(M,r,1,0.)[1]);sys.exit()
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 4000;Sf=int(sys.argv[3]) if len(sys.argv)>3 else 1200;out={};t0=time.time();r=np.random.default_rng(sd+3)
    M=make(sd);h=fit(M,lambda:text(r)[0],S,2e-3);out['pre']=dict(hist=h,nll=nll(M,7),ev=evaluate(M,900+sd,1,False),can_read=can_read(M,800+sd),can_copy=can_copy(M,850+sd));print(sd,'预训练',h[-1],out['pre']['ev'],'前缀可读',out['pre']['can_read'],'抄笔记',out['pre']['can_copy'],round(time.time()-t0),flush=True)
    if Sf==0:sys.exit()
    # 对照：只在原文本上继续微调
    Mcc=clone(M);fit(Mcc,lambda:text(r)[0],Sf,5e-4);out['cc']=dict(nll=nll(Mcc,7),ev=evaluate(Mcc,900+sd,1,False))
    res={}
    for tau in (0.,.5,1.,2.):
        Cs,st,allv=annotate(M,np.random.default_rng(sd+10),800 if tau==.5 else 200,tauf=tau);st['precision']=st['kept_ok']/max(1,st['kept']);res[str(tau)]=st;print(sd,'过滤',tau,st,round(time.time()-t0),flush=True)
        if tau==.5:   # 论文里计算器也用 τ=0.5：数据少，答案也短，差值本来就小
            Cbest=Cs;vals=np.array([d for d,g in allv]);gd=np.array([g for d,g in allv]);out['dist']=dict(good=np.histogram(vals[gd],bins=np.arange(-3,8,.5))[0].tolist(),bad=np.histogram(vals[~gd],bins=np.arange(-3,8,.5))[0].tolist())
    out['filter']=res;out['cstar']=dict(total=len(Cbest),with_call=res['0.5']['kept'])
    Tf=clone(M);n=len(Cbest);fit(Tf,lambda:Cbest[int(r.integers(0,n))],Sf,5e-4);out['tf']={};out['tf_nll']=nll(Tf,7)
    for k in (0,1,3,10):
        out['tf'][str(k)]=evaluate(Tf,900+sd,max(k,1),k>0);print(sd,'Toolformer k',k,out['tf'][str(k)],round(time.time()-t0),flush=True)
    json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
    np.savez(f'm{sd}.npz',**{f'p{i}':p.v for i,p in enumerate(Tf.ps)})
    out['calls']=call_stats(Tf,900+sd);print(sd,'调用写得对不对',out['calls'],flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
