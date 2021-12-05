'''
Running jobs in multiple processes.
'''

from __future__ import annotations
from pathlib import Path
import multiprocessing
import multiprocessing.pool  # needed for type annotations
from typing import NamedTuple, Callable, Sequence, Mapping, Optional, Any

from .lib_io import RedirectIO
from .lib_logging import ProgressBar


class Job(NamedTuple):
    name: str
    func: Callable
    args: Sequence = ()
    kwargs: Mapping = {}
    stdout: Optional[Path] = None
    stderr: Optional[Path] = None

class JobResult(NamedTuple):
    job: Job
    result: Any
    worker: str

def run_jobs_with_multiprocessing(jobs: Sequence[Job], n_processes: Optional[int] = None, progress_bar: bool = False, 
        callback: Optional[Callable[[JobResult], Any]] = None, pool: Optional[multiprocessing.pool.Pool] = None) -> list[JobResult]:
    '''Run jobs (i.e. call job.func(*job.args, **job.kwargs)) in n_processes processes. 
    Standard output and standard error output are saved in files job.stdout and job.stderr.
    Default n_processes: number of CPUs.
    If n_processes==1, then run jobs sequentially without starting new processes (useful for debugging).'''
    if n_processes is None and pool is None:
        n_processes = multiprocessing.cpu_count()
    n_jobs = len(jobs)
    results = []
    with ProgressBar(n_jobs, title=f'Running {n_jobs} jobs in {n_processes} processes', mute = not progress_bar) as bar:
        if pool is not None:
            result_iterator = pool.imap_unordered(_run_job, jobs)
            for result in result_iterator:
                if callback is not None:
                    callback(result)
                results.append(result)
                bar.step()
        elif n_processes == 1:
            for job in jobs:
                result = _run_job(job)
                if callback is not None:
                    callback(result)
                results.append(result)
                bar.step()
        else:
            with multiprocessing.Pool(n_processes) as ad_hoc_pool:
                result_iterator = ad_hoc_pool.imap_unordered(_run_job, jobs)
                for result in result_iterator:
                    if callback is not None:
                        callback(result)
                    results.append(result)
                    bar.step()
    return results
    
def _run_job(job: Job) -> JobResult:
    worker = multiprocessing.current_process().name
    with RedirectIO(stdout=job.stdout, stderr=job.stderr):
        result = job.func(*job.args, **job.kwargs)
    return JobResult(job=job, result=result, worker=worker)

