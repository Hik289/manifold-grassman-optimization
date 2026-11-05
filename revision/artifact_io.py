"""Atomic figure publishing for a live LaTeX preview reading the same files."""
import os
from pathlib import Path


def save_figure(fig,path,**kwargs):
    path=Path(path)
    temp=path.with_name(f'.{path.stem}.{os.getpid()}.partial{path.suffix}')
    fig.savefig(temp,**kwargs)
    temp.replace(path)
