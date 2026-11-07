#!/usr/bin/env python3
"""Read-only evaluation of saved CIFAR-10 development prototype banks."""
import csv, hashlib, json, pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT=Path.home()/"Projects/SOM_CIFAR10_NewStudy"
SPLIT=ROOT/'design/splits_v1/indices.npz'
PCA=ROOT/'runs/pca64_dev_v1/pca64.npz'
OUT=ROOT/'runs/development_evaluation_v1'
METHODS=('som','winner','random','kmeans++')
EXPECTED_SPLIT='17c764251b013c1d09ef4fe5ccff301db48c8c71f08b4d1e1022cd9d931dec97'
EXPECTED_PCA='60e110d9755119ef94d34a275eba38bf601d1a026eead04317a1dadac8c476a7'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def assign(x,w):
    # Work in chunks to avoid large temporary matrices.
    result=[]; sums=[]
    ww=np.einsum('ij,ij->i',w,w)
    for start in range(0,len(x),512):
        a=x[start:start+512]
        d=np.maximum(np.einsum('ij,ij->i',a,a)[:,None]+ww[None,:]-2*a@w.T,0)
        p=np.argmin(d,axis=1)
        result.append(p)
        sums.append(d[np.arange(len(a)),p])
    return np.concatenate(result),float(np.mean(np.concatenate(sums)))
def main():
    assert sha(SPLIT)==EXPECTED_SPLIT, 'split hash mismatch'
    assert sha(PCA)==EXPECTED_PCA, 'PCA hash mismatch'
    with np.load(SPLIT,allow_pickle=False) as z:
        ids=z['draw_00_n1000'].copy(); dev=z['development_indices'].copy()
    assert len(dev)==5000 and not np.intersect1d(ids,dev).size
    chunks=[]; labels=[]
    for i in range(1,6):
        with (ROOT/'data/raw/cifar-10-batches-py'/f'data_batch_{i}').open('rb') as f:
            b=pickle.load(f,encoding='bytes')
        chunks.append(b[b'data']);labels.append(np.asarray(b[b'labels']))
    raw=np.concatenate(chunks); y=np.concatenate(labels)[dev]
    rgb=raw[dev].astype(np.float32)/255.0
    del raw,chunks,labels
    with np.load(PCA,allow_pickle=False) as z:
        assert np.array_equal(z['dev_ids'],dev)
        pca=z['dev64'].astype(np.float32).copy()
    rows=[]; evidence={}
    for rep,folder in [('centered_rgb','four_arm_dev_v1'),('pca64','four_arm_pca64_dev_v1')]:
        path=ROOT/'runs'/folder
        events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
        logs={e['method']:e for e in events if e.get('event')=='arm_complete'}
        assert set(logs)==set(METHODS)
        for method in METHODS:
            f=path/(method.replace('+','p')+'.npz')
            assert sha(f)==logs[method]['sha256'],f'hash mismatch: {f}'
            with np.load(f,allow_pickle=False) as z:
                w=z['final_weights'].astype(np.float32).copy()
                saved_ids=z['train_ids'].copy()
                center=z['train_mean'].astype(np.float32).copy()
            assert np.array_equal(saved_ids,ids)
            x=rgb-center if rep=='centered_rgb' else pca
            assert w.shape==(16,x.shape[1]) and np.isfinite(w).all()
            pred,mse=assign(x,w)
            row={'representation':rep,'method':method,'development_images':len(dev),
                 'quantization_mse':mse,'ari':float(adjusted_rand_score(y,pred)),
                 'nmi':float(normalized_mutual_info_score(y,pred)),
                 'occupied_prototypes':int(np.unique(pred).size)}
            rows.append(row);evidence[f'{rep}/{method}']=sha(f)
            print(json.dumps(row,sort_keys=True),flush=True)
    assert len(rows)==8
    OUT.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0])
    with (OUT/'development_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    manifest={'status':'development_only','dataset':'CIFAR-10 training batches only',
              'split_sha256':sha(SPLIT),'pca_sha256':sha(PCA),
              'source_model_sha256':evidence,'results_sha256':sha(OUT/'development_metrics.csv')}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DEVELOPMENT EVALUATION: PASS')
    print('RESULTS:',OUT/'development_metrics.csv')
if __name__=='__main__': main()
