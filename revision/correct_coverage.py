"""Stop only this revision's workers and preserve superseded pilot tasks."""
import json
import os
from pathlib import Path
import shutil
import signal
import time

here=Path(__file__).resolve().parent
launch=json.loads((here/'launch.json').read_text())
for worker in launch['workers']:
    pid=worker['pid']
    proc=Path(f'/proc/{pid}/cmdline')
    if not proc.exists():continue
    cmd=proc.read_bytes().replace(b'\x00',b' ').decode()
    if '/codex_revision_20261101/revision/suite.py' not in cmd:
        raise RuntimeError(f'Refusing to signal unrelated PID {pid}')
    os.kill(pid,signal.SIGTERM)
    print('Stopped own worker',pid,flush=True)
time.sleep(3)
archive=here/'exploratory_archive/initial_coverage_pilots'
archive.mkdir(parents=True,exist_ok=True)
for name in ['lowrank_128','lowrank_256','lowrank_512','recsys_movielens']:
    src=here/'results'/name
    if src.exists():
        dst=archive/name
        if dst.exists():raise RuntimeError(f'Archive already exists: {dst}')
        shutil.move(str(src),str(dst))
        print('Archived pilot',name,flush=True)
shutil.copy2(here/'launch.json',archive/'initial_launch.json')
