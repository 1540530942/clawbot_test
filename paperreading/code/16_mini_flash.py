#!/usr/bin/env python3
# 第 16 期：只依赖 NumPy，直接运行 python3 16_mini_flash.py
"""第 16 期：不用 GPU 也能验证的三件事。
1 分块算出的结果和直接算完全一样；2 按算法 0、算法 1 数 HBM 读写，看 N、d、M 的影响；
3 CPU 上实测：直接算（存下 N×N 矩阵）对比分块算的时间和峰值内存。CPU 没有 GPU 那样的 SRAM/HBM 差异，只作参考。"""
import json,time,tracemalloc,math,numpy as np
def direct(Q,K,V):
    S=Q@K.T;S=S-S.max(1,keepdims=True);P=np.exp(S);P/=P.sum(1,keepdims=True);return P@V
def flash(Q,K,V,Bc,Br):
    N,d=Q.shape;O=np.zeros((N,d));l=np.zeros(N);m=np.full(N,-np.inf)
    for j in range(0,N,Bc):
        Kj,Vj=K[j:j+Bc],V[j:j+Bc]
        for i in range(0,N,Br):
            S=Q[i:i+Br]@Kj.T;mt=S.max(1);P=np.exp(S-mt[:,None]);lt=P.sum(1);mn=np.maximum(m[i:i+Br],mt)
            a=np.exp(m[i:i+Br]-mn)*l[i:i+Br];b=np.exp(mt-mn);ln=a+b*lt
            O[i:i+Br]=((a)[:,None]*O[i:i+Br]+b[:,None]*(P@Vj))/ln[:,None];l[i:i+Br]=ln;m[i:i+Br]=mn
    return O
def io_std(N,d):return dict(read=2*N*d+N*N+N*N+N*d,write=N*N+N*N+N*d)
def io_flash(N,d,M):
    Bc=max(1,math.ceil(M/(4*d)));Bc=min(Bc,N);Tc=math.ceil(N/Bc)
    return dict(read=2*N*d+Tc*(N*d+N*d+2*N),write=Tc*(N*d+2*N),Bc=Bc,Tc=Tc)
tot=lambda x:x['read']+x['write']
out={}
r=np.random.default_rng(0);errs=[]
for N,d,Bc,Br in((64,8,8,8),(256,32,16,32),(1024,64,100,64)):
    Q,K,V=[r.normal(size=(N,d))/math.sqrt(d) for _ in range(3)];errs.append((N,d,Bc,Br,float(np.abs(direct(Q,K,V)-flash(Q,K,V,Bc,Br)).max())))
out['exact']=errs;print(errs)
out['io']=[(N,d,M,tot(io_std(N,d)),tot(io_flash(N,d,M)),io_flash(N,d,M)['Bc'],io_flash(N,d,M)['Tc']) for N,d,M in [(n,64,96000) for n in(128,256,512,1024,2048,4096,8192,16384,32768,65536)]+[(1024,d,96000) for d in(16,32,64,128,256,512)]+[(1024,64,m) for m in(2000,4000,8000,16000,32000,64000,96000,192000)]+[(1024,512,2000),(256,512,2000)]]
for x in out['io']:print(x,round(x[3]/x[4],2))
tm=[]
for N,Bc in((1024,256),(2048,256),(4096,256)):
    d=64;Q,K,V=[r.normal(size=(N,d))/8 for _ in range(3)];row=dict(N=N)
    for name,fn in(('direct',lambda:direct(Q,K,V)),('flash',lambda:flash(Q,K,V,Bc,64))):
        t=time.time();fn();a=time.time()-t;t=time.time();fn();a=min(a,time.time()-t);tracemalloc.start();fn();pk=tracemalloc.get_traced_memory()[1];tracemalloc.stop();row[name]=(round(a*1000,1),round(pk/1e6,1))
    tm.append(row);print(row,flush=True)
out['cpu']=tm
json.dump(out,open('res.json','w'))
