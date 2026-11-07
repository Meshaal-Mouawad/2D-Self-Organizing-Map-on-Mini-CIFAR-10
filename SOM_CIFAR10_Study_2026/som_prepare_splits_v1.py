#!/usr/bin/env python3
"""Create immutable candidate CIFAR-10 development and paired training indices.
No test data is opened. Original data and prior pilots remain untouched.
"""
import hashlib, json, pickle, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'design' / 'splits_v1'
DATA = ROOT / 'data' / 'raw' / 'cifar-10-batches-py'
SEED = 20261107
DRAWS = 10

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if OUT.exists():
        raise SystemExit(f'EXISTS: {OUT}. Refusing to overwrite an existing split. No changes made.')
    labels=[]
    for i in range(1,6):
        with (DATA / f'data_batch_{i}').open('rb') as f:
            b=pickle.load(f, encoding='bytes')
        labels.extend(b[b'labels'])
    y=np.asarray(labels, dtype=np.int16)
    assert y.shape == (50000,)
    assert np.array_equal(np.bincount(y), np.full(10,5000))
    rng=np.random.default_rng(SEED)
    dev=[]; pool=[]
    for c in range(10):
        ids=np.flatnonzero(y==c)
        ids=rng.permutation(ids)
        dev.extend(ids[:500].tolist()); pool.extend(ids[500:].tolist())
    dev=np.asarray(sorted(dev),dtype=np.int32)
    pool=np.asarray(sorted(pool),dtype=np.int32)
    assert len(dev)==5000 and len(pool)==45000
    assert len(np.intersect1d(dev,pool))==0
    assert np.array_equal(np.bincount(y[dev]), np.full(10,500))
    assert np.array_equal(np.bincount(y[pool]), np.full(10,4500))
    splits={'development_indices':dev,'training_pool_indices':pool}
    for draw in range(DRAWS):
        small=[]; large=[]
        for c in range(10):
            ids=rng.permutation(pool[y[pool]==c])
            small.extend(ids[:100].tolist()); large.extend(ids[:500].tolist())
        small=np.asarray(sorted(small),dtype=np.int32)
        large=np.asarray(sorted(large),dtype=np.int32)
        assert len(small)==1000 and len(large)==5000
        assert np.isin(small,large).all()
        assert np.array_equal(np.bincount(y[small]),np.full(10,100))
        assert np.array_equal(np.bincount(y[large]),np.full(10,500))
        splits[f'draw_{draw:02d}_n1000']=small
        splits[f'draw_{draw:02d}_n5000']=large
    OUT.mkdir(parents=True)
    path=OUT/'indices.npz'
    np.savez(path,**splits)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(), 'status':'candidate_not_frozen',
        'dataset':'official CIFAR-10 training batches only','seed':SEED,'draws':DRAWS,
        'development_size':5000,'training_pool_size':45000,'budgets':[1000,5000],
        'python':sys.version.split()[0], 'numpy':np.__version__, 'platform':platform.platform(),
        'index_semantics':'zero-based indices into concatenated official data_batch_1..5',
        'sha256_indices_npz':sha(path)}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DEVELOPMENT:',len(dev),'TRAINING POOL:',len(pool))
    print('PAIRED DRAWS:',DRAWS,'BUDGETS: 1000, 5000')
    print('INDEX FILE:',path)
    print('SHA256:',manifest['sha256_indices_npz'])
    print('TEST PARTITION: NOT OPENED')
    print('SPLIT PREPARATION: PASS')
if __name__=='__main__': main()
