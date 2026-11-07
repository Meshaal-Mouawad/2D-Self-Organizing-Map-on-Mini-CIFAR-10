#!/usr/bin/env python3
"""Development-only four-arm SOM initialization benchmark. Never opens CIFAR test data.
Fixed: draw 00, n=1000, PCA-64, start 0. This is NOT a confirmatory run.
"""
import hashlib, json, pickle, time, platform, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs'/'four_arm_pca64_dev_v1'
PCA=ROOT/'runs'/'pca64_dev_v1'/'pca64.npz'
PCA_SHA='60e110d9755119ef94d34a275eba38bf601d1a026eead04317a1dadac8c476a7'
SPLIT=ROOT/'design'/'splits_v1'/'indices.npz'
EXPECTED='17c764251b013c1d09ef4fe5ccff301db48c8c71f08b4d1e1022cd9d931dec97'
SEED=20261107; K=16; EPOCHS=5

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def log(obj):
    obj={'timestamp_utc':datetime.now(timezone.utc).isoformat(),**obj}
    with (OUT/'events.jsonl').open('a') as f: f.write(json.dumps(obj,sort_keys=True)+'\n')
    print(json.dumps(obj,sort_keys=True),flush=True)

def sqdist(x,w):
    d=np.sum(x*x,axis=1)[:,None]+np.sum(w*w,axis=1)[None,:]-2*x@w.T
    return np.maximum(d,0)

def lloyd(x,initial,max_iter=100):
    w=initial.copy(); trace=[]
    for it in range(max_iter):
        d=sqdist(x,w); labels=np.argmin(d,axis=1)
        new=w.copy()
        counts=np.bincount(labels,minlength=K)
        for j in range(K):
            if counts[j]: new[j]=np.mean(x[labels==j],axis=0)
        # Deterministic farthest-residual reseeding for empty clusters.
        used=set()
        residual=d[np.arange(len(x)),labels]
        for j in np.flatnonzero(counts==0):
            for idx in np.argsort(-residual,kind='stable'):
                if int(idx) not in used:
                    new[j]=x[idx]; used.add(int(idx)); break
        shift=np.linalg.norm(new-w)/(np.linalg.norm(w)+1e-12)
        w=new
        trace.append(float(shift))
        if shift<1e-6: break
    d=sqdist(x,w); labels=np.argmin(d,axis=1)
    return w,labels,trace,float(np.mean(d[np.arange(len(x)),labels]))

def warm(x,initial,order,method):
    w=initial.copy(); grid=np.indices((4,4)).reshape(2,-1).T
    for t,idx in enumerate(order):
        frac=t/max(len(order)-1,1)
        eta=.5*(.05/.5)**frac
        sigma=2*(.5/2)**frac
        v=x[idx]; b=int(np.argmin(np.sum((w-v)**2,axis=1)))
        if method=='som':
            g=np.exp(-np.sum((grid-grid[b])**2,axis=1)/(2*sigma*sigma))
        else:
            g=np.zeros(K);g[b]=1
        w+=eta*g[:,None]*(v-w)
        if not np.isfinite(w).all(): raise ValueError('nonfinite weights')
    return w

def main():
    if sha(SPLIT)!=EXPECTED: raise RuntimeError('Split checksum mismatch')
    if OUT.exists(): raise SystemExit('Development output exists; refusing to overwrite: '+str(OUT))
    with np.load(SPLIT,allow_pickle=False) as z:
        ids=z['draw_00_n1000'].copy(); dev=z['development_indices'].copy()
    assert len(ids)==1000 and not np.intersect1d(ids,dev).size
    if sha(PCA)!=PCA_SHA: raise RuntimeError('PCA artifact checksum mismatch')
    with np.load(PCA,allow_pickle=False) as z:
        assert np.array_equal(z['train_ids'],ids)
        assert np.array_equal(z['dev_ids'],dev)
        x=z['train64'].astype(np.float32).copy()
    assert x.shape==(1000,64) and np.isfinite(x).all()
    # Read training labels solely for a development-only diagnostic.
    chunks=[]
    for i in range(1,6):
        with (ROOT/'data'/'raw'/'cifar-10-batches-py'/f'data_batch_{i}').open('rb') as f:
            b=pickle.load(f,encoding='bytes')
        chunks.append(np.asarray(b[b'labels'],dtype=np.int64))
    y=np.concatenate(chunks)
    del chunks
    mean=np.zeros((1,64),dtype=np.float32) # PCA scores are already centered.
    rng=np.random.default_rng(SEED)
    initial_ids=rng.choice(len(x),K,replace=False)
    initial=x[initial_ids].copy()
    order=np.concatenate([rng.permutation(len(x)) for _ in range(EPOCHS)])
    OUT.mkdir(parents=True)
    config={'status':'development_only','subset':'draw_00_n1000','representation':'pca64','seed':SEED,'epochs':EPOCHS,'prototypes':K,'split_sha256':EXPECTED,'pca_sha256':PCA_SHA,'python':sys.version.split()[0],'numpy':np.__version__,'platform':platform.platform(),'lloyd_max_iter':100,'warm_schedule':'eta .5->.05; sigma 2->.5'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    log({'event':'start',**config})
    for method in ('som','winner','random','kmeans++'):
        start=time.perf_counter()
        if method in ('som','winner'):
            bank=warm(x,initial,order,method)
        elif method=='random': bank=initial.copy()
        else:
            bank,_=kmeans_plusplus(x,n_clusters=K,random_state=SEED)
        init_time=time.perf_counter()-start
        start=time.perf_counter()
        w,pred,trace,mse=lloyd(x,bank)
        refine_time=time.perf_counter()-start
        if not np.isfinite(w).all() or not np.isfinite(mse): raise RuntimeError('Nonfinite result')
        output=OUT/f'{method.replace("+","p")}.npz'
        np.savez(output,initial_bank=bank,final_weights=w,train_assignments=pred,train_ids=ids,lloyd_shifts=np.array(trace),train_mean=mean)
        log({'event':'arm_complete','method':method,'initialization_seconds':round(init_time,4),'refinement_seconds':round(refine_time,4),'train_quantization_mse':mse,'train_ari_diagnostic':float(adjusted_rand_score(y[ids],pred)),'occupied_prototypes':int(len(np.unique(pred))),'lloyd_iterations':len(trace),'output':str(output),'sha256':sha(output)})
    log({'event':'complete','arms':4,'confirmatory_runs':0})
    print('PCA-64 FOUR-ARM DEVELOPMENT BENCHMARK: PASS')

if __name__=='__main__': main()
