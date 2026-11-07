#!/usr/bin/env python3
"""Read-only protocol validation and one isolated development smoke test.
Never starts or modifies confirmatory runs. Requires existing local study files.
"""
import hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import adjusted_rand_score
from som_four_arm_development_v1 import warm, lloyd

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'design/protocol_v1/confirmatory_protocol.json'
EXPECTED='f4460874e272bf0288b049ce10a5a21f2f8ee02197ba8c8a417cfd85be9dc479'
SPLIT=ROOT/'design/splits_v1/indices.npz'
SPLIT_SHA='17c764251b013c1d09ef4fe5ccff301db48c8c71f08b4d1e1022cd9d931dec97'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert sha(PROTOCOL)==EXPECTED,'Protocol hash mismatch'
    assert sha(SPLIT)==SPLIT_SHA,'Split hash mismatch'
    c=json.loads(PROTOCOL.read_text())
    assert c['confirmatory_runs']==320 and c['completed_confirmatory_runs']==0
    assert c['lloyd_max_iterations']==300
    assert c['training_epochs']==5 and c['som_grid']==[4,4]
    assert c['methods']==['som','winner','random','kmeans++']
    assert c['training_budgets']==[1000,5000]
    assert c['representations']==['centered_rgb','pca64']
    assert c['draws']==list(range(10)) and c['starts']==[0,1]
    runs=[(n,rep,draw,start,method) for n in c['training_budgets']
          for rep in c['representations'] for draw in c['draws']
          for start in c['starts'] for method in c['methods']]
    assert len(runs)==len(set(runs))==320
    with np.load(SPLIT,allow_pickle=False) as z:
        dev=z['development_indices'].copy()
        ids=z['draw_00_n1000'].copy()
        for draw in range(10):
            small=z[f'draw_{draw:02d}_n1000']
            large=z[f'draw_{draw:02d}_n5000']
            assert len(small)==1000 and len(large)==5000
            assert np.isin(small,large).all() and not np.intersect1d(large,dev).size
    seeds=[20261107+draw*100+start for draw in range(10) for start in range(2)]
    assert len(set(seeds))==20
    print('PROTOCOL AND 320-ARM PLAN: PASS',flush=True)
    print('ALL NESTED SPLITS AND 20 START SEEDS: PASS',flush=True)
    print('=== ISOLATED DEVELOPMENT SMOKE TEST ===',flush=True)
    chunks=[]; labels=[]
    for i in range(1,6):
        with (ROOT/'data/raw/cifar-10-batches-py'/f'data_batch_{i}').open('rb') as f:
            b=pickle.load(f,encoding='bytes')
        chunks.append(b[b'data']);labels.extend(b[b'labels'])
    raw=np.concatenate(chunks)
    x=raw[ids].astype(np.float32)/255.0
    x-=x.mean(axis=0,keepdims=True)
    rng=np.random.default_rng(seeds[0])
    initial=x[rng.choice(len(x),16,replace=False)].copy()
    order=np.concatenate([rng.permutation(len(x)) for _ in range(5)])
    t=time.perf_counter()
    bank=warm(x,initial,order,'som')
    w,pred,trace,mse=lloyd(x,bank,max_iter=300)
    elapsed=time.perf_counter()-t
    assert w.shape==(16,3072) and np.isfinite(w).all()
    assert len(trace)<=300 and np.isfinite(mse)
    print(json.dumps({'smoke_method':'som','sample_size':len(ids),'iterations':len(trace),
                      'train_mse':mse,'train_ari':float(adjusted_rand_score(np.asarray(labels)[ids],pred)),
                      'elapsed_seconds':round(elapsed,4)},sort_keys=True),flush=True)
    print('CONTROLLER VALIDATION SMOKE: PASS',flush=True)
    print('CONFIRMATORY RUNS EXECUTED BY THIS SCRIPT: 0',flush=True)

if __name__=='__main__':
    main()
