#!/usr/bin/env python3
"""Three-draw, four-arm, two-representation development reproducibility benchmark.
No CIFAR test data. Outputs are development-only, not confirmatory evidence.
Resume at completed arm boundaries; never overwrite completed arm files.
"""
import csv, hashlib, json, os, pickle, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from som_four_arm_development_v1 import warm, lloyd, sqdist

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs'/'multiseed_dev_v1'
SPLIT=ROOT/'design'/'splits_v1'/'indices.npz'
SPLIT_SHA='17c764251b013c1d09ef4fe5ccff301db48c8c71f08b4d1e1022cd9d931dec97'
METHODS=('som','winner','random','kmeans++')
FIELDS=['representation','draw','method','seed','train_n','dev_n','train_mse','dev_mse','dev_ari','dev_nmi','dev_occupied','initialization_seconds','refinement_seconds','pca_seconds','artifact_sha256']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def event(**kwargs):
    entry={'timestamp_utc':datetime.now(timezone.utc).isoformat(),**kwargs}
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(entry,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(entry,sort_keys=True),flush=True)

def load_training():
    images=[];labels=[]
    for i in range(1,6):
        with (ROOT/'data/raw/cifar-10-batches-py'/f'data_batch_{i}').open('rb') as f:
            b=pickle.load(f,encoding='bytes')
        images.append(b[b'data']);labels.extend(b[b'labels'])
    return np.concatenate(images),np.asarray(labels)

def evaluate(x,w,y):
    d=sqdist(x,w); pred=np.argmin(d,axis=1)
    return float(np.mean(d[np.arange(len(x)),pred])),float(adjusted_rand_score(y,pred)),float(normalized_mutual_info_score(y,pred)),int(len(np.unique(pred)))

def main():
    assert sha(SPLIT)==SPLIT_SHA,'Split checksum mismatch'
    OUT.mkdir(parents=True,exist_ok=True)
    cfg={'status':'development_only','draws':[0,1,2],'train_n':1000,'dev_n':5000,'representations':['centered_rgb','pca64'],'methods':list(METHODS),'epochs':5,'prototypes':16,'pca_components':64,'split_sha256':SPLIT_SHA,'python':sys.version.split()[0],'numpy':np.__version__,'platform':platform.platform()}
    cfg_path=OUT/'config.json'
    if cfg_path.exists(): assert json.loads(cfg_path.read_text())==cfg,'Configuration changed: refusing resume'
    else: cfg_path.write_text(json.dumps(cfg,indent=2)+'\n')
    with np.load(SPLIT,allow_pickle=False) as z:
        dev_ids=z['development_indices'].copy()
        draws=[z[f'draw_{i:02d}_n1000'].copy() for i in range(3)]
    raw,y=load_training()
    records=[]
    event(event='invocation_start',total_arms=24)
    for draw,ids in enumerate(draws):
        assert len(ids)==1000 and not np.intersect1d(ids,dev_ids).size
        train=raw[ids].astype(np.float32)/255.0
        dev=raw[dev_ids].astype(np.float32)/255.0
        mu=train.mean(axis=0,keepdims=True)
        train-=mu;dev-=mu
        for rep in ('centered_rgb','pca64'):
            if rep=='pca64':
                start=time.perf_counter()
                pca=PCA(n_components=64,svd_solver='randomized',random_state=20261107+draw)
                x=pca.fit_transform(train).astype(np.float32)
                v=pca.transform(dev).astype(np.float32)
                pca_time=time.perf_counter()-start
            else: x,v=train,dev;pca_time=0.0
            seed=20261107+draw
            rng=np.random.default_rng(seed)
            initial=x[rng.choice(len(x),16,replace=False)].copy()
            order=np.concatenate([rng.permutation(len(x)) for _ in range(5)])
            for method in METHODS:
                path=OUT/f'draw{draw:02d}_{rep}_{method.replace("+","p")}.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:
                        w=z['final_weights'].copy(); init_sec=float(z['initialization_seconds']); refine_sec=float(z['refinement_seconds']);train_mse=float(z['train_mse'])
                    assert w.shape==(16,x.shape[1]) and np.isfinite(w).all()
                    status='reused'
                else:
                    start=time.perf_counter()
                    if method in ('som','winner'): bank=warm(x,initial,order,method)
                    elif method=='random':bank=initial.copy()
                    else:bank,_=kmeans_plusplus(x,n_clusters=16,random_state=seed)
                    init_sec=time.perf_counter()-start
                    start=time.perf_counter()
                    w,pred,trace,train_mse=lloyd(x,bank)
                    refine_sec=time.perf_counter()-start
                    assert w.shape==(16,x.shape[1]) and np.isfinite(w).all()
                    tmp=path.with_name(path.stem+'.tmp.npz')
                    np.savez_compressed(tmp,final_weights=w,initial_bank=bank,train_ids=ids,train_mse=train_mse,initialization_seconds=init_sec,refinement_seconds=refine_sec,lloyd_shifts=np.asarray(trace))
                    os.replace(tmp,path)
                    status='created'
                dev_mse,ari,nmi,occupied=evaluate(v,w,y[dev_ids])
                record={'representation':rep,'draw':draw,'method':method,'seed':seed,'train_n':1000,'dev_n':5000,'train_mse':train_mse,'dev_mse':dev_mse,'dev_ari':ari,'dev_nmi':nmi,'dev_occupied':occupied,'initialization_seconds':init_sec,'refinement_seconds':refine_sec,'pca_seconds':pca_time,'artifact_sha256':sha(path)}
                records.append(record)
                with (OUT/'results.csv.tmp').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(records)
                os.replace(OUT/'results.csv.tmp',OUT/'results.csv')
                event(event='arm_complete',status=status,draw=draw,representation=rep,method=method,completed=len(records),total=24,dev_ari=ari,dev_mse=dev_mse,sha256=record['artifact_sha256'])
    assert len(records)==24
    event(event='complete',completed=24,total=24,confirmatory_runs=0)
    print('MULTISEED DEVELOPMENT: PASS',flush=True)

if __name__=='__main__':main()
