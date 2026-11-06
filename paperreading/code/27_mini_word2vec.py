#!/usr/bin/env python3
# 从头实现 CBOW、Skip-gram、Huffman 树，和 LSA、迷你 NNLM 对比
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

"""第 27 期玩具实验：手写 CBOW、Skip-gram（Huffman 分层 softmax），和 LSA、迷你 NNLM 比较。
两份语料：① 模板生成的合成语料，有标准类比题；② 本栏目论文全文（真实文本，只看近邻）。
合成语料不代表自然语言，类比分数不能和论文比。"""
import sys,re,json,time,heapq,glob,collections,numpy as np
PAIRS={'capital':[('france','paris'),('germany','berlin'),('italy','rome'),('spain','madrid'),('japan','tokyo'),('china','beijing'),('russia','moscow'),('egypt','cairo'),('greece','athens'),('norway','oslo'),('canada','ottawa'),('india','delhi'),('kenya','nairobi'),('peru','lima'),('cuba','havana'),('iran','tehran')],
'gender':[('king','queen'),('man','woman'),('boy','girl'),('brother','sister'),('father','mother'),('son','daughter'),('uncle','aunt'),('husband','wife'),('prince','princess'),('actor','actress'),('hero','heroine'),('waiter','waitress')],
'comparative':[(a,a+s) for a,s in [('tall','er'),('small','er'),('old','er'),('fast','er'),('cold','er'),('long','er'),('short','er'),('strong','er'),('quick','er'),('young','er'),('rich','er'),('kind','er')]],
'past':[(a,a+'ed') for a in ['walk','jump','talk','work','play','cook','clean','paint','watch','visit','help','call']],
'plural':[(a,a+'s') for a in ['cat','dog','bird','book','tree','car','house','river','farmer','doctor','teacher','student']]}
SEMANTIC=('capital','gender');SYNTACTIC=('comparative','past','plural')
TOP=['ancient','famous','quiet','busy','green','sunny','snowy','wide','modern','small','northern','southern','coastal','wild','lovely','poor']
# 每对词有自己的话题词。第一次运行没有这一层，语法类比的两个词毫无共现线索，分数只等于随机猜
T_CMP=['mountain','insect','castle','train','winter','road','story','horse','fox','puppy','banker','nurse']
T_PST=['park','fence','phone','office','guitar','kitchen','floor','wall','movie','museum','neighbor','taxi']
T_PLU=['barn','leash','nest','shelf','forest','garage','roof','bridge','field','clinic','school','campus']
def sent(r):
    g=lambda L:L[int(r.integers(0,len(L)))]
    u=r.random();cats=PAIRS;N=[x for x,_ in cats['plural']];Np=[y for _,y in cats['plural']]
    if u<.25:
        c,p=g(cats['capital']);i=[x for x,_ in cats['capital']].index(c);t=TOP[i];T=[f'the capital of {c} is {p}',f'{p} is a {t} city in {c}',f'{c} is a {t} country and {p} is its capital',f'people visit the city of {p}',f'people visit the country of {c}',f'the {t} streets of {p}',f'the {t} hills of {c}'];return g(T)
    if u<.5:
        m,f=g(cats['gender']);i=[x for x,_ in cats['gender']].index(m);t=TOP[i];T=[f'he is a {t} {m} and she is a {t} {f}',f'the {m} said that he was tired',f'the {f} said that she was tired',f'the {t} {m} and his {g(N)}',f'the {t} {f} and her {g(N)}',f'a {m} walks with a {f}',f'he became a {m}',f'she became a {f}'];return g(T)
    if u<.65:
        a,c=g(cats['comparative']);z=T_CMP[[x for x,_ in cats['comparative']].index(a)];T=[f'the {z} is {a}',f'the {z} is very {a}',f'the {z} is {c} than the {g(N)}',f'this {z} is much {c} than that one',f'a {a} {z}',f'how {a} it is'];return g(T)
    if u<.8:
        v,d=g(cats['past']);n=g(N);z=T_PST[[x for x,_ in cats['past']].index(v)];T=[f'every day they {v} near the {z}',f'they often {v} by the {z}',f'yesterday they {d} near the {z}',f'last year we {d} by the {z}',f'the {n} will {v}',f'the {n} {d} away'];return g(T)
    n,p=g(cats['plural']);z=T_PLU[[x for x,_ in cats['plural']].index(n)];T=[f'one {n} is at the {z}',f'this {n} is old',f'a {n} sat by the {z}',f'two {p} are at the {z}',f'these {p} are old',f'many {p} sat by the {z}'];return g(T)
def synth(n,seed):
    r=np.random.default_rng(seed);o=[]
    while sum(map(len,o))<n:o.append(sent(r).split())
    return o
def questions():
    Q={}
    for k,P in PAIRS.items():
        Q[k]=[(a,b,c,d) for i,(a,b) in enumerate(P) for j,(c,d) in enumerate(P) if i!=j]
    return Q
class Huff:
    def __init__(s,cnt):
        h=[(c,i,None,None) for i,c in enumerate(cnt)];heapq.heapify(h);k=len(cnt);s.V=len(cnt)
        while len(h)>1:
            a=heapq.heappop(h);b=heapq.heappop(h);heapq.heappush(h,(a[0]+b[0],k,a,b));k+=1
        s.code=[None]*s.V;s.pt=[None]*s.V
        stack=[(h[0],[],[])]
        while stack:
            n,c,p=stack.pop()
            if n[2] is None:s.code[n[1]]=np.array(c,float);s.pt[n[1]]=np.array(p,int);continue
            nid=n[1]-s.V;stack.append((n[2],c+[0],p+[nid]));stack.append((n[3],c+[1],p+[nid]))
class W2V:
    def __init__(s,sents,D,mincount=1,seed=0):
        c=collections.Counter(w for x in sents for w in x);s.w=[w for w,n in c.most_common() if n>=mincount];s.id={w:i for i,w in enumerate(s.w)};s.cnt=np.array([c[w] for w in s.w]);s.D=D
        s.data=[np.array([s.id[w] for w in x if w in s.id]) for x in sents];r=np.random.default_rng(seed);s.r=r;s.In=(r.random((len(s.w),D))-.5)/D;s.Out=np.zeros((len(s.w),D));s.h=Huff(s.cnt);s.tot=sum(map(len,s.data))
    def hs(s,v,t,lr):
        L=s.Out[s.h.pt[t]];f=1/(1+np.exp(-np.clip(L@v,-20,20)));g=(1-s.h.code[t]-f)*lr;s.Out[s.h.pt[t]]+=np.outer(g,v);return g@L
    def train(s,mode,E=3,C=None,lr0=.025):
        C=C or (4 if mode=='cbow' else 5);n=0;total=s.tot*E;t0=time.time();r=s.r
        for ep in range(E):
            for x in s.data:
                for i,t in enumerate(x):
                    lr=max(lr0*(1-n/total),lr0*1e-4);n+=1
                    if mode=='cbow':
                        a=max(0,i-C);b=i+C+1;ctx=np.r_[x[a:i],x[i+1:b]]
                        if len(ctx)==0:continue
                        h=s.In[ctx].mean(0);e=s.hs(h,t,lr);s.In[ctx]+=e
                    else:
                        R=int(r.integers(1,C+1))
                        for j in range(max(0,i-R),min(len(x),i+R+1)):
                            if j!=i:
                                e=s.hs(s.In[t],x[j],lr);s.In[t]+=e
        return n/(time.time()-t0)
def lsa(sents,D,C=4):
    c=collections.Counter(w for x in sents for w in x);W=[w for w,_ in c.most_common()];I={w:i for i,w in enumerate(W)};M=np.zeros((len(W),len(W)))
    for x in sents:
        for i,w in enumerate(x):
            for j in range(max(0,i-C),min(len(x),i+C+1)):
                if j!=i:M[I[w],I[x[j]]]+=1
    tot=M.sum();pw=M.sum(1,keepdims=True)/tot;P=np.maximum(np.log(np.maximum(M/tot,1e-12)/(pw*pw.T)),0);U,S,_=np.linalg.svd(P,full_matrices=False);return W,U[:,:D]*np.sqrt(S[:D])
def nnlm(sents,D,N=4,H=64,steps=None,seed=0):
    """迷你前馈语言模型（Bengio 2003）：前 N 个词 → 隐藏层 → 全词表 softmax"""
    c=collections.Counter(w for x in sents for w in x);W=[w for w,_ in c.most_common()];I={w:i for i,w in enumerate(W)};V=len(W);r=np.random.default_rng(seed);E=r.normal(0,.1,(V,D));A=r.normal(0,.1,(N*D,H));Bo=r.normal(0,.1,(H,V));b=np.zeros(H);bo=np.zeros(V);X=[];Y=[]
    for x in sents:
        ids=[I[w] for w in x];p=[0]*N+ids
        for i in range(N,len(p)):X.append(p[i-N:i]);Y.append(p[i])
    X=np.array(X);Y=np.array(Y);steps=steps or 3*len(X)//64;t0=time.time()
    for st in range(steps):
        k=r.integers(0,len(X),64);e=E[X[k]].reshape(64,-1);h=np.tanh(e@A+b);lg=h@Bo+bo;lg-=lg.max(1,keepdims=True);q=np.exp(lg);q/=q.sum(1,keepdims=True);d=q.copy();d[np.arange(64),Y[k]]-=1;dh=d@Bo.T;dtanh=dh*(1-h*h);lr=.5*(1-st/steps)+1e-3;Bo-=lr*h.T@d/64;bo-=lr*d.sum(0)/64;A-=lr*e.T@dtanh/64;b-=lr*dtanh.sum(0)/64;de=(dtanh@A.T).reshape(64,N,D);np.add.at(E,X[k],-lr*de/64)
    return W,E,64*steps/(time.time()-t0)
def analogy(W,E,Q,exclude=True,maxq=400,seed=0):
    I={w:i for i,w in enumerate(W)};V=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9);r=np.random.default_rng(seed);res={}
    for k,qs in Q.items():
        qs=[q for q in qs if all(x in I for x in q)]
        if len(qs)>maxq:qs=[qs[i] for i in r.choice(len(qs),maxq,replace=False)]
        ok=0
        for a,b,c,d in qs:
            y=V[I[b]]-V[I[a]]+V[I[c]];s=V@y
            if exclude:s[[I[a],I[b],I[c]]]=-9
            ok+=W[int(s.argmax())]==d
        res[k]=ok/max(1,len(qs))
    res['semantic']=float(np.mean([res[k] for k in SEMANTIC]));res['syntactic']=float(np.mean([res[k] for k in SYNTACTIC]));res['total']=float(np.mean([res[k] for k in PAIRS]));return res
STOP=set('the of and to a in is that for we on with as are this be by it from an at can our which have not or was was has these their each than were but its also more one two all if used use using such where there other between when both may will they into only then been while however see over so than per most some different'.split())
def real(minc=8):
    S=[]
    for f in sorted(glob.glob('../../papers/*/paper.txt')):
        t=open(f,encoding='utf-8',errors='ignore').read().lower();cut=t.find('references');t=t if cut<0 else t
        for line in re.split(r'[.!?\n]+',t):
            w=re.findall(r"[a-z][a-z\-]+",line)
            if len(w)>=4:S.append(w)
    return S
if __name__=='__main__':
    sd=int(sys.argv[1]);Nt=int(sys.argv[2]) if len(sys.argv)>2 else 400000
    out={};Q=questions();t0=time.time()
    corp=synth(Nt,sd)
    for D in (32,):
        for mode in('cbow','sg'):
            m=W2V(corp,D,seed=sd);sp=m.train(mode);out[f'{mode}/{D}']=dict(an=analogy(m.w,m.In,Q),an_inc=analogy(m.w,m.In,Q,exclude=False),speed=sp,sec=round(time.time()-t0));print(sd,mode,D,out[f'{mode}/{D}']['an'],round(sp),flush=True)
            if mode=='sg' and sd==0:np.savez('m_syn.npz',E=m.In,W=np.array(m.w));
        W,E=lsa(corp,D);out[f'lsa/{D}']=dict(an=analogy(W,E,Q));print(sd,'lsa',out[f'lsa/{D}']['an'],flush=True)
        W,E,sp=nnlm(corp,D,seed=sd);out[f'nnlm/{D}']=dict(an=analogy(W,E,Q),speed=sp);print(sd,'nnlm',out[f'nnlm/{D}']['an'],round(sp),round(time.time()-t0),flush=True)
        json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
    # 表 2 的形状：维度 × 数据量（CBOW）
    tb={}
    for n in (Nt//16,Nt//4,Nt):
        c2=synth(n,sd+1)
        for D in (4,8,16,32,64):
            m=W2V(c2,D,seed=sd);m.train('cbow');tb[f'{n}/{D}']=analogy(m.w,m.In,Q)['total'];print(sd,'tb',n,D,tb[f'{n}/{D}'],flush=True)
    out['tb']=tb;json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
    if sd==0:
        S=real();m=W2V(S,48,mincount=8,seed=0);m.train('sg',E=2);n=m.cnt/m.cnt.sum();H=float(-(n*np.log2(n)).sum());al=float((n*np.array([len(c) for c in m.h.code])).sum());out['real']=dict(V=len(m.w),tok=int(m.tot),logV=float(np.log2(len(m.w))),H=H,huff=al);keep=[i for i,w in enumerate(m.w) if w not in STOP][:1500];np.savez('m_real.npz',E=m.In[keep],W=np.array([m.w[i] for i in keep]),C=m.cnt[keep]);print(out['real'],round(time.time()-t0),flush=True)
        json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
