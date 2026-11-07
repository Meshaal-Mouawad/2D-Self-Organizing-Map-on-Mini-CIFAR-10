#!/usr/bin/env python3
"""Resumable local SOM/CIFAR-10 experiment controller. No network or test-set access.
Default is read-only plan. --run --max-runs 1 executes one arm; --run --all executes remaining arms.
Completed arm files are immutable and SHA256 verified on resume. Results rebuilt from arm records.
"""
import argparse, csv, hashlib, json, os, pickle, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from som_four_arm_development_v1 import warm, lloyd, sqdist

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'design/protocol_v1/confirmatory_protocol.json'
EXPECTED='f4460874e272bf0288b049ce10a5a21f2f8ee02197ba8c8a417cfd85be9dc479'
OUT=ROOT/'runs/confirmatory_v1'
METHODS=('som','winner','random','kmeans++')
FIELDS=('run_id','budget','representation','draw','start','method','seed','train_mse','dev_mse','dev_ari','dev_nmi','dev_occupied','lloyd_iterations','initialization_seconds','refinement_seconds','pca_seconds','artifact_sha256')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write_atomic(path,data):
    tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'w') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def log(**fields):
    obj={'timestamp_utc':datetime.now(timezone.utc).isoformat(),**fields}
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(obj,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(obj,sort_keys=True),flush=True)

def plan():
    return [(n,rep,d,s,m) for n in (1000,5000) for rep in ('centered_rgb','pca64') for d in range(10) for s in range(2) for m in METHODS]

def rid(t):
    n,rep,d,s,m=t
    return f'n{n}_{rep}_d{d:02d}_s{s}_{m.replace("+","p")}'

def read_config():
    assert sha(PROTOCOL)==EXPECTED,'Frozen protocol SHA256 mismatch'
    c=json.loads(PROTOCOL.read_text())
    assert c['confirmatory_runs']==320 and c['lloyd_max_iterations']==300
    assert c['preprocessing']['pixel_normalization']=='float32 / 255.0'
    assert c['preprocessing']['split_sha256']==sha(ROOT/'design/splits_v1/indices.npz')
    assert len(plan())==320 and len({rid(t) for t in plan()})==320
    return c

def existing_records():
    records={}
    for p in sorted((OUT/'arms').glob('*.json')):
        r=json.loads(p.read_text()); key=r['run_id']
        assert key not in records and key==p.stem,'Duplicate/invalid arm record'
        artifact=OUT/'arms'/f'{key}.npz'
        assert artifact.is_file() and sha(artifact)==r['artifact_sha256'],f'Corrupt artifact: {key}'
        assert all(np.isfinite(float(r[k])) for k in ('train_mse','dev_mse','dev_ari','dev_nmi'))
        records[key]=r
    return records

def rebuild(records):
    rows=[records[rid(t)] for t in plan() if rid(t) in records]
    import io
    s=io.StringIO(); w=csv.DictWriter(s,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    write_atomic(OUT/'results.csv',s.getvalue())

def load_raw():
    images=[]; labels=[]
    for i in range(1,6):
        with (ROOT/'data/raw/cifar-10-batches-py'/f'data_batch_{i}').open('rb') as f:
            b=pickle.load(f,encoding='bytes')
        images.append(b[b'data']);labels.extend(b[b'labels'])
    return np.concatenate(images),np.asarray(labels)

def eval_dev(x,w,y):
    preds=[]; residual=[]
    for i in range(0,len(x),512):
        d=sqdist(x[i:i+512],w); p=np.argmin(d,axis=1)
        preds.append(p);residual.append(d[np.arange(len(p)),p])
    p=np.concatenate(preds)
    return float(np.mean(np.concatenate(residual))),float(adjusted_rand_score(y,p)),float(normalized_mutual_info_score(y,p)),int(np.unique(p).size)

def execute(args):
    c=read_config(); OUT.mkdir(parents=True,exist_ok=True);(OUT/'arms').mkdir(exist_ok=True)
    lock=OUT/'controller.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError: raise RuntimeError(f'Lock exists: {lock}. Check process before manual removal.')
    os.write(fd,f'pid={os.getpid()} utc={datetime.now(timezone.utc).isoformat()}\n'.encode());os.close(fd)
    try:
        records=existing_records(); rebuild(records)
        pending=[t for t in plan() if rid(t) not in records]
        log(event='invocation_start',completed=len(records),total=320,pending=len(pending),protocol_sha256=EXPECTED)
        if not pending:return
        raw,y=load_raw()
        with np.load(ROOT/'design/splits_v1/indices.npz',allow_pickle=False) as z:
            dev_ids=z['development_indices'].copy()
            splits={(d,n):z[f'draw_{d:02d}_n{n}'].copy() for d in range(10) for n in (1000,5000)}
        assert len(dev_ids)==5000
        limit=len(pending) if args.all else args.max_runs
        last_group=None
        for t in pending[:limit]:
            n,rep,draw,start,method=t; key=rid(t)
            group=(n,rep,draw)
            if group!=last_group:
                ids=splits[draw,n]
                assert len(ids)==n and len(np.unique(ids))==n and not np.intersect1d(ids,dev_ids).size
                x=raw[ids].astype(np.float32)/255.0
                v=raw[dev_ids].astype(np.float32)/255.0
                mu=x.mean(axis=0,keepdims=True);x-=mu;v-=mu
                pca_seconds=0.0
                if rep=='pca64':
                    tick=time.perf_counter()
                    pca=PCA(n_components=64,svd_solver='randomized',random_state=20261107+draw)
                    xx=pca.fit_transform(x).astype(np.float32);vv=pca.transform(v).astype(np.float32)
                    x,v=xx,vv;pca_seconds=time.perf_counter()-tick
                last_group=group
            seed=20261107+draw*100+start
            rng=np.random.default_rng(seed)
            initial=x[rng.choice(len(x),16,replace=False)].copy()
            order=np.concatenate([rng.permutation(len(x)) for _ in range(5)])
            tick=time.perf_counter()
            if method in ('som','winner'):bank=warm(x,initial,order,method)
            elif method=='random':bank=initial.copy()
            else:bank,_=kmeans_plusplus(x,n_clusters=16,random_state=seed)
            init_seconds=time.perf_counter()-tick
            tick=time.perf_counter();w,pred,trace,train_mse=lloyd(x,bank,max_iter=300)
            refine_seconds=time.perf_counter()-tick
            assert w.shape==(16,x.shape[1]) and np.isfinite(w).all() and np.isfinite(train_mse)
            dm,ari,nmi,occ=eval_dev(v,w,y[dev_ids])
            path=OUT/'arms'/f'{key}.npz'; tmp=path.with_name(path.stem+'.tmp.npz')
            np.savez_compressed(tmp,final_weights=w,initial_bank=bank,train_ids=ids,train_mse=train_mse,lloyd_shifts=np.asarray(trace),seed=seed)
            os.replace(tmp,path)
            r=dict(run_id=key,budget=n,representation=rep,draw=draw,start=start,method=method,seed=seed,train_mse=float(train_mse),dev_mse=dm,dev_ari=ari,dev_nmi=nmi,dev_occupied=occ,lloyd_iterations=len(trace),initialization_seconds=init_seconds,refinement_seconds=refine_seconds,pca_seconds=pca_seconds,artifact_sha256=sha(path))
            write_atomic(OUT/'arms'/f'{key}.json',json.dumps(r,sort_keys=True,indent=2)+'\n')
            records[key]=r;rebuild(records)
            log(event='arm_complete',run_id=key,completed=len(records),total=320,dev_ari=ari,dev_mse=dm,sha256=r['artifact_sha256'])
        log(event='invocation_complete',completed=len(records),total=320,remaining=320-len(records))
    finally:
        lock.unlink(missing_ok=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true',help='Enable execution; default only checks protocol and plan')
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--max-runs',type=int,default=1,help='Number of pending arms to execute (default 1)')
    group.add_argument('--all',action='store_true',help='Execute all remaining arms')
    args=ap.parse_args()
    read_config()
    if not args.run:
        count=len(existing_records()) if (OUT/'arms').exists() else 0
        print(f'PLAN VALIDATION: PASS | completed={count}/320 | pending={320-count} | execution=OFF')
        return
    assert args.max_runs>=1
    execute(args)

if __name__=='__main__':main()
