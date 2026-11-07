#!/usr/bin/env python3
"""Controlled interruption/restart test of the existing SOM pilot, isolated outputs."""
from pathlib import Path
import hashlib, subprocess, sys
root=Path.cwd()
source=root/'som_cifar10_pilot_v1.py'
assert source.is_file(), f'Missing {source}'
original=source.read_text()
assert "OUT = ROOT / 'runs/pilot_v1'" in original
assert 'SEED, PER_CLASS, EPOCHS, SIDE = 20261107, 20, 2, 4' in original
assert "event('checkpoint', completed_epochs=epoch+1" in original
out=root/'runs/pilot_interrupt_test'
assert not out.exists(), f'Refusing to overwrite existing test: {out}'
out.mkdir(parents=True)
variant=original.replace("OUT = ROOT / 'runs/pilot_v1'", "OUT = ROOT / 'runs/pilot_interrupt_test'")
variant=variant.replace('SEED, PER_CLASS, EPOCHS, SIDE = 20261107, 20, 2, 4', 'SEED, PER_CLASS, EPOCHS, SIDE = 20261107, 20, 3, 4')
needle="              elapsed_seconds=round(time.monotonic()-start, 3),\n              checkpoint_path=str(CHECKPOINT), checkpoint_sha256=sha256(CHECKPOINT))"
assert needle in variant
interrupted=variant.replace(needle,needle+"\n        if epoch == 0:\n            raise SystemExit('CONTROLLED INTERRUPTION AFTER EPOCH 1')")
first=out/'interrupted_worker.py'
second=out/'resumed_worker.py'
first.write_text(interrupted)
second.write_text(variant)
print('=== PHASE A: CONTROLLED INTERRUPTION ===',flush=True)
a=subprocess.run([sys.executable,str(first)],cwd=root)
assert a.returncode != 0, 'Expected controlled interruption'
cp=out/'checkpoint.npz'
assert cp.exists(), 'No checkpoint created'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
h1=digest(cp)
import numpy as np
with np.load(cp,allow_pickle=False) as d:
    assert int(d['completed_epochs'])==1
    assert np.isfinite(d['weights']).all()
print('CHECKPOINT AFTER INTERRUPTION:',h1,flush=True)
print('=== PHASE B: RESUME ===',flush=True)
b=subprocess.run([sys.executable,str(second)],cwd=root)
assert b.returncode == 0, f'Resume failed: {b.returncode}'
with np.load(cp,allow_pickle=False) as d:
    assert int(d['completed_epochs'])==3
    assert np.isfinite(d['weights']).all()
h2=digest(cp)
print('FINAL CHECKPOINT:',h2,flush=True)
print('=== PHASE C: COMPLETED RESTART ===',flush=True)
c=subprocess.run([sys.executable,str(second)],cwd=root)
assert c.returncode == 0
assert digest(cp)==h2, 'Completed restart changed checkpoint'
print('INTERRUPTED-RUN RECOVERY: PASS',flush=True)
print('Evidence folder:',out,flush=True)
