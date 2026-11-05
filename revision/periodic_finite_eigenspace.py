"""Repair the observed CPU eigensolver NaN without redoing a parameter step.

The captured failure has finite G, C, U, M, V and W, but non-finite eigenvectors
returned by eigh(C). At that exact pre-transport failure point, finish the
refresh with the mathematically equivalent thin SVD of G. Other errors abort.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import strength_graph_rescue as recovery
from strength_study import PeriodicDiag
from grassroute import _polar_orthogonal


class FinitePeriodicDiag(PeriodicDiag):
    @torch.no_grad()
    def _step_routed(self,group,p):
        try:return super()._step_routed(group,p)
        except torch.linalg.LinAlgError as error:
            tb=error.__traceback__;failure=None
            while tb:
                if tb.tb_frame.f_code.co_name=='_step_routed' and tb.tb_frame.f_code.co_filename.endswith('baselines.py'):
                    failure=tb.tb_frame.f_locals
                tb=tb.tb_next
            if failure is None:raise
            required=['G','C','U','U_new','M','V']
            if any(k not in failure for k in required):raise
            if torch.isfinite(failure['U_new']).all():raise
            if not all(torch.isfinite(failure[k]).all() for k in ['G','C','U','M','V']):raise
            if not torch.isfinite(p).all():raise
            # Parent has already updated W,M,V and incremented ages. Its failed
            # polar decomposition precedes ALL moment transport and U assignment.
            g,u=failure['G'],failure['U'];st=self.state[p];r=group['rank']
            if st['U'].data_ptr()!=u.data_ptr():raise
            vh=torch.linalg.svd(g,full_matrices=False).Vh
            unew=vh[:r].T.contiguous();rotation=_polar_orthogonal(unew.T@u)
            st['M'].copy_(st['M']@rotation.T);st['V'].copy_(st['V']@rotation.T.square())
            st['U']=unew
            st['diag']['nonfinite_eigh_fallbacks']=st['diag'].get('nonfinite_eigh_fallbacks',0)+1


def main():
    original=recovery.original;root=original.OUT
    protocol=dict(task='graph_citeseer',method='PeriodicDiag',
        original_grid=original.candidates('PeriodicDiag'),seed=42,screening_epochs=100,formal_epochs=1000,
        change='Only finite-input/non-finite-eigenvector CPU backend failures: finish refresh with thin SVD of G',
        no_double_parameter_update=True,first_and_second_moment_transport=True,
        evidence='periodic_full_failure_diagnostic: epoch 275, finite G/C/U/M/V/W and 75% finite U_new',
        test_selection=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    recovery.atomic(root/'graph_finite_eigenspace_protocol.json',protocol)
    original.PeriodicDiag=FinitePeriodicDiag;original.dump=recovery.atomic
    folder=root/'graph/graph_citeseer/PeriodicDiag/backend_repair'
    candidates=[]
    for i,hp in enumerate(original.candidates('PeriodicDiag')):
        result=recovery.safe_run('graph_citeseer','PeriodicDiag',hp,100,folder/f'screen_{i}',False)
        candidates.append(result);print('BACKEND-REPAIRED SCREEN',hp,result['status'],result.get('validation'),flush=True)
    good=sorted([x for x in candidates if x['status']=='complete'],key=lambda x:x['validation']['score'],reverse=True)
    for attempt,best in enumerate(good):
        dest=folder/f'formal_attempt_{attempt}'
        result=recovery.safe_run('graph_citeseer','PeriodicDiag',best['hp'],1000,dest,True)
        if result['status']!='complete':continue
        import shutil
        canonical=folder.parent/'formal';canonical.mkdir(exist_ok=True)
        shutil.copy2(dest/'trajectory.json',canonical/'trajectory.json')
        result=dict(result,execution_policy='finite-input-eigensolver-fallback-v1',source_manifest='graph_finite_eigenspace_protocol.json')
        recovery.atomic(canonical/'summary.json',result)
        recovery.atomic(folder.parent/'selection.json',dict(hp=best['hp'],criterion='maximum final screening-validation accuracy',
            candidates=[dict(hp=x['hp'],status=x['status'],validation=x.get('validation')) for x in candidates],
            policy='graph_finite_eigenspace_protocol.json'))
        recovery.atomic(folder.parent/'recovered.json',dict(status='complete',prior_failures_retained=True,
            backend_fallback='thin SVD only after finite inputs produce non-finite eigh eigenvectors'))
        recovery.aggregate();print('BACKEND REPAIRED FORMAL COMPLETE',result,flush=True);return
    recovery.atomic(folder/'failed.json',dict(reason='No repaired candidate completed',prior_failures_retained=True))


if __name__=='__main__':main()
