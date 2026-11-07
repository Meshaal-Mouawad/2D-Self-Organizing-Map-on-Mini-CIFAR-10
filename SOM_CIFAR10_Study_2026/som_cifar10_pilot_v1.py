#!/usr/bin/env python3
"""Development-only SOM smoke test; no test data used. Run from project root."""
import hashlib, json, os, pickle, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
DATA = ROOT / 'data/raw/cifar-10-batches-py'
OUT = ROOT / 'runs/pilot_v1'
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / 'events.jsonl'
CHECKPOINT = OUT / 'checkpoint.npz'
SEED, PER_CLASS, EPOCHS, SIDE = 20261107, 20, 2, 4

def event(kind, **fields):
    item = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'event': kind, **fields}
    with LOG.open('a', encoding='utf8') as f:
        f.write(json.dumps(item, sort_keys=True) + '\n')
    print(json.dumps(item, sort_keys=True), flush=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def load_training():
    xs, ys = [], []
    for i in range(1, 6):
        with (DATA / f'data_batch_{i}').open('rb') as f:
            batch = pickle.load(f, encoding='bytes')
        xs.append(batch[b'data'])
        ys.append(np.asarray(batch[b'labels']))
    return np.concatenate(xs), np.concatenate(ys)

def main():
    if not DATA.is_dir():
        raise FileNotFoundError(f'Missing extracted dataset: {DATA}')
    xall, yall = load_training()
    assert xall.shape == (50000, 3072) and yall.shape == (50000,)
    rng = np.random.default_rng(SEED)
    ids = np.concatenate([rng.choice(np.flatnonzero(yall == c), PER_CLASS, replace=False) for c in range(10)])
    rng.shuffle(ids)
    x = xall[ids].astype(np.float32) / np.float32(255)
    x -= x.mean(axis=0, keepdims=True)
    assert np.isfinite(x).all()
    coords = np.stack(np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing='ij'), axis=-1).reshape(-1,2)
    grid_d2 = ((coords[:,None,:] - coords[None,:,:])**2).sum(axis=2)
    if CHECKPOINT.exists():
        with np.load(CHECKPOINT) as cp:
            w = cp['weights'].copy()
            epoch_start = int(cp['completed_epochs'])
            saved_seed = int(cp['seed'])
            saved_ids = cp['ids']
        if saved_seed != SEED or not np.array_equal(saved_ids, ids):
            raise RuntimeError('Checkpoint does not match pilot configuration/subset')
        event('resume', completed_epochs=epoch_start, checkpoint_sha256=sha256(CHECKPOINT))
    else:
        w = x[rng.choice(len(x), SIDE*SIDE, replace=False)].copy()
        epoch_start = 0
        event('start', seed=SEED, samples=len(x), classes=10, prototypes=len(w), epochs=EPOCHS,
              python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(),
              source='official CIFAR-10 training batches only', representation='mean-centered RGB pixels')
    start = time.monotonic()
    for epoch in range(epoch_start, EPOCHS):
        order = np.random.default_rng(SEED + 1000 + epoch).permutation(len(x))
        for step, idx in enumerate(order):
            t = epoch*len(x)+step
            fraction = t/max(1, EPOCHS*len(x)-1)
            eta = 0.35*(1-fraction)+0.05*fraction
            sigma = 2.0*(1-fraction)+0.6*fraction
            dist = ((w-x[idx])**2).sum(axis=1)
            bmu = int(np.argmin(dist))
            h = np.exp(-grid_d2[:,bmu]/(2*sigma*sigma)).astype(np.float32)
            w += np.float32(eta)*h[:,None]*(x[idx]-w)
            if not np.isfinite(w).all():
                raise FloatingPointError(f'Nonfinite prototypes at epoch {epoch}, step {step}')
        distances = ((x[:,None,:]-w[None,:,:])**2).sum(axis=2)
        assignments = distances.argmin(axis=1)
        qe = float(distances.min(axis=1).mean())
        occupancy = int(np.unique(assignments).size)
        tmp = OUT / 'checkpoint.tmp.npz'
        np.savez_compressed(tmp, weights=w, completed_epochs=epoch+1, seed=SEED, ids=ids)
        os.replace(tmp, CHECKPOINT)
        event('checkpoint', completed_epochs=epoch+1, total_epochs=EPOCHS,
              quantization_mse=qe, occupied_prototypes=occupancy,
              elapsed_seconds=round(time.monotonic()-start, 3),
              checkpoint_path=str(CHECKPOINT), checkpoint_sha256=sha256(CHECKPOINT))
    event('complete', completed_epochs=EPOCHS, total_epochs=EPOCHS,
          elapsed_seconds_this_invocation=round(time.monotonic()-start,3),
          checkpoint_sha256=sha256(CHECKPOINT))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        event('failure', error_type=type(exc).__name__, message=str(exc))
        raise
