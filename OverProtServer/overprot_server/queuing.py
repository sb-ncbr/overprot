import os
import time
import rq  # type: ignore
import redis
from enum import Enum
from pathlib import Path
import subprocess
import shutil
import json
from contextlib import suppress
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Union

from .constants import DB_DIR_PENDING, DB_DIR_RUNNING, DB_DIR_COMPLETED, DB_DIR_FAILED, DB_DIR_ARCHIVED, DB_DIR_DELETED, JOB_STATUSINFO_FILE, JOB_INFO_FILE, JOB_DOMAINS_FILE, JOB_DATADIR, JOB_DATAZIP, JOB_RESULT_FILE, JOB_STDOUT_FILE, JOB_STDERR_FILE, JOB_ERROR_MESSAGE_FILE, OVERPROT_PYTHON, OVERPROT_PY, TIME_FORMAT, COMPLETED_JOB_STORING_DAYS, QUEUE_NAME, JOB_TIMEOUT, JOB_CLEANUP_TIMEOUT
from .domain_parsing import Domain


def enqueue_job(job_name: str, domains: List[Domain], info: dict) -> Tuple['JobStatusInfo', Any]:
    job_status = JobStatusInfo(name=job_name)  # default status Pending
    Path(DB_DIR_PENDING, job_status.id).mkdir(parents=True, exist_ok=True)
    job_status.save(Path(DB_DIR_PENDING, job_status.id, JOB_STATUSINFO_FILE))
    Path(DB_DIR_PENDING, job_status.id, JOB_DOMAINS_FILE).write_text('\n'.join(str(domain) for domain in domains))
    Path(DB_DIR_PENDING, job_status.id, JOB_INFO_FILE).write_text(json.dumps(info))

    queue = rq.Queue(QUEUE_NAME, connection=redis.Redis.from_url('redis://'), default_timeout=JOB_TIMEOUT+JOB_CLEANUP_TIMEOUT)
    job = queue.enqueue(process_job, job_status.id)
    queue_id = job.get_id()
    return job_status, queue_id
    
def process_job(job_id: str):
    # time.sleep(10)
    move(Path(DB_DIR_PENDING, job_id), Path(DB_DIR_RUNNING, job_id))
    job_status = JobStatusInfo.update(Path(DB_DIR_RUNNING, job_id, JOB_STATUSINFO_FILE), status=JobStatus.RUNNING, start_time=get_current_time())
    assert job_status.id == job_id

    info = json.loads(Path(DB_DIR_RUNNING, job_id, JOB_INFO_FILE).read_text())
    with Path(DB_DIR_RUNNING, job_id, JOB_RESULT_FILE).open('w') as w:
        for k, v in info.items():
            w.write(f'{k}: {v}\n\n')
        w.write('Running...\n\n')

    # time.sleep(10)
    try:
        proc = subprocess.run([OVERPROT_PYTHON, OVERPROT_PY, job_id, 'all', Path(DB_DIR_RUNNING, job_id, JOB_DATADIR), 
                            '--domains', Path(DB_DIR_RUNNING, job_id, JOB_DOMAINS_FILE)], 
                            timeout=JOB_TIMEOUT, capture_output=True, encoding='utf8')
        proc_returncode: Optional[int] = proc.returncode
        proc_stdout = proc.stdout
        proc_stderr = proc.stderr
        successfull = proc.returncode==0
        Path(DB_DIR_RUNNING, job_id, 'pica.txt').write_text(f'code: {proc_returncode}')
    except subprocess.TimeoutExpired as ex:
        proc_returncode = None
        proc_stdout = anystr_to_str(ex.stdout)
        proc_stderr = anystr_to_str(ex.stderr)
        successfull = False
        Path(DB_DIR_RUNNING, job_id, JOB_ERROR_MESSAGE_FILE).write_text(f'Job timeout expired ({JOB_TIMEOUT} seconds).')
    except Exception as ex:
        proc_returncode = None
        proc_stdout = ''
        proc_stderr = str(ex)
        successfull = False
        error_message = f"Unexpected error:\n{type(ex).__name__}: {ex}"
        Path(DB_DIR_RUNNING, job_id, JOB_ERROR_MESSAGE_FILE).write_text(error_message)


    with Path(DB_DIR_RUNNING, job_id, JOB_RESULT_FILE).open('a') as w:
        if successfull:
            w.write('Completed\n\n')
        elif proc_returncode is not None: 
            w.write(f'Failed (exit code: {proc_returncode})\n\n')
        else: 
            w.write(f'Failed (timed out)\n\n')
        w.write(f'Stdout:\n{proc_stdout}\n\n')
        w.write(f'Stderr:\n{proc_stderr}\n\n')
    Path(DB_DIR_RUNNING, job_id, JOB_STDOUT_FILE).write_text(proc_stdout)
    Path(DB_DIR_RUNNING, job_id, JOB_STDERR_FILE).write_text(proc_stderr)

    if not successfull and proc_returncode is not None:
        err_msg = extract_error_message(proc_stderr)
        Path(DB_DIR_RUNNING, job_id, JOB_ERROR_MESSAGE_FILE).write_text(err_msg)
        


    if successfull:
        # TODO select only important results
        shutil.copy(Path(DB_DIR_RUNNING, job_id, JOB_STDOUT_FILE), Path(DB_DIR_RUNNING, job_id, JOB_DATADIR, JOB_STDOUT_FILE))
        shutil.copy(Path(DB_DIR_RUNNING, job_id, JOB_STDERR_FILE), Path(DB_DIR_RUNNING, job_id, JOB_DATADIR, JOB_STDERR_FILE))
        make_archive(Path(DB_DIR_RUNNING, job_id, JOB_DATADIR), Path(DB_DIR_RUNNING, job_id, JOB_DATAZIP))
        # shutil.make_archive(Path(DB_DIR_RUNNING, job_id, JOB_DATAZIP), 'zip', Path(DB_DIR_RUNNING, job_id, JOB_DATADIR))
        # move(Path(DB_DIR_RUNNING, job_id, JOB_DATAZIP+'.zip'), Path(DB_DIR_RUNNING, job_id, JOB_DATAZIP))
        move(Path(DB_DIR_RUNNING, job_id, JOB_DATADIR, 'results'), Path(DB_DIR_RUNNING, job_id, 'results'))
        make_archive(Path(DB_DIR_RUNNING, job_id, 'results'), Path(DB_DIR_RUNNING, job_id, 'results.zip'))
        # shutil.make_archive(Path(DB_DIR_RUNNING, job_id, 'results'), 'zip', Path(DB_DIR_RUNNING, job_id, 'results'))
        shutil.rmtree(Path(DB_DIR_RUNNING, job_id, JOB_DATADIR))

    end_time = get_current_time()
    delete_time = end_time + timedelta(COMPLETED_JOB_STORING_DAYS)
    if successfull:
        Path(DB_DIR_COMPLETED).mkdir(parents=True, exist_ok=True)
        move(Path(DB_DIR_RUNNING, job_id), Path(DB_DIR_COMPLETED, job_id))
        job_status = JobStatusInfo.update(Path(DB_DIR_COMPLETED, job_id, JOB_STATUSINFO_FILE), status=JobStatus.COMPLETED, end_time=end_time, delete_time=delete_time)
    else:
        Path(DB_DIR_FAILED).mkdir(parents=True, exist_ok=True)
        move(Path(DB_DIR_RUNNING, job_id), Path(DB_DIR_FAILED, job_id))
        job_status = JobStatusInfo.update(Path(DB_DIR_FAILED, job_id, JOB_STATUSINFO_FILE), status=JobStatus.FAILED, end_time=end_time, delete_time=delete_time)

def anystr_to_str(string: Union[str, bytes]) -> str:
    if isinstance(string, str):
        return string
    elif isinstance(string, bytes):
        try:
            return string.decode('utf8')
        except UnicodeDecodeError:
            return str(string)
    else:
        return str(string)

def extract_error_message(stderr: str) -> str:
    error_lines = [line.strip() for line in stderr.splitlines() if line.strip() != '']
    if len(error_lines) > 0:
        last_line = error_lines[-1]
        if ':' in last_line:
            return last_line.split(':', maxsplit=1)[1].strip()
        else:
            return last_line
    else:
        return ''


def move(src: Path, dest: Path) -> Path:
    return Path(shutil.move(str(src), str(dest)))

def make_archive(src: Path, dest: Path) -> Path:
    fmt = dest.suffix.lstrip('.')
    archive = shutil.make_archive(str(dest.with_suffix('')), fmt, str(src))
    return Path(archive)

def get_current_time() -> datetime:
    return datetime.utcnow().replace(microsecond=0)

def parse_time(time: str) -> Optional[datetime]:
    if time == '':
        return None
    return datetime.strptime(time, TIME_FORMAT)

def format_time(time: Optional[datetime]) -> str:
    if time is None:
        return ''
    return datetime.strftime(time, TIME_FORMAT)
    
def get_uuid() -> str:
    return str(uuid.uuid4())


class JobStatus(Enum):
    NONEXISTENT = 'Nonexistent'  # never existed
    PENDING = 'Pending'  # in queue
    RUNNING = 'Running'  # taken from queue, being processed
    COMPLETED = 'Completed'  # processed succesfully
    FAILED = 'Failed'  # processed with error
    ARCHIVED = 'Archived' # protected against deletion
    DELETED = 'Deleted'  # processed, maximum storing period has elapsed and data have been removed

@dataclass
class JobStatusInfo(object):
    id: str = field(default_factory=get_uuid)
    name: str = ''
    status: JobStatus = JobStatus.PENDING
    submission_time: datetime = field(default_factory=get_current_time)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    delete_time: Optional[datetime] = None    

    @staticmethod
    def load(filename: Path) -> 'JobStatusInfo':
        d = {}
        with open(filename) as r:
            for line in r:
                line = line.strip()
                if line != '':
                    key, value = line.split('=', maxsplit=1)
                    d[key] = value
        submission_time=parse_time(d['submission_time'])
        assert submission_time is not None
        result = JobStatusInfo(id=d['id'], 
                               name=d['name'], 
                               status=JobStatus(d['status']),
                               submission_time=submission_time,
                               start_time=parse_time(d['start_time']),
                               end_time=parse_time(d['submission_time']), 
                               delete_time=parse_time(d['delete_time']))
        return result
    
    def save(self, filename: Path) -> None:
        with open(filename, 'w') as w:
            w.write(f'id={self.id}\n')
            w.write(f'name={self.name}\n')
            w.write(f'status={self.status.value}\n')
            w.write(f'submission_time={format_time(self.submission_time)}\n')
            w.write(f'start_time={format_time(self.start_time)}\n')
            w.write(f'end_time={format_time(self.end_time)}\n')
            w.write(f'delete_time={format_time(self.delete_time)}\n')

    @staticmethod
    def update(filename: Path, status: Optional[JobStatus] = None, submission_time: Optional[datetime] = None,
               start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, delete_time: Optional[datetime] = None) -> 'JobStatusInfo':
        '''Update job status info in file filename and return the updated JobStatusInfo.'''
        info = JobStatusInfo.load(filename)
        if status is not None:
            info.status = status
        if submission_time is not None:
            info.submission_time = submission_time
        if start_time is not None:
            info.start_time = start_time
        if end_time is not None:
            info.end_time = end_time
        if delete_time is not None:
            info.delete_time = delete_time
        info.save(filename)
        return info
    
    @staticmethod
    def load_for_job(job_id: str) -> 'JobStatusInfo':
        try:
            return JobStatusInfo.load(Path(DB_DIR_PENDING, job_id, JOB_STATUSINFO_FILE))
        except FileNotFoundError: pass
        try:
            return JobStatusInfo.load(Path(DB_DIR_RUNNING, job_id, JOB_STATUSINFO_FILE))
        except FileNotFoundError: pass
        try:
            return JobStatusInfo.load(Path(DB_DIR_COMPLETED, job_id, JOB_STATUSINFO_FILE))
        except FileNotFoundError: pass
        try:
            return JobStatusInfo.load(Path(DB_DIR_FAILED, job_id, JOB_STATUSINFO_FILE))
        except FileNotFoundError: pass
        try:
            return JobStatusInfo.load(Path(DB_DIR_ARCHIVED, job_id, JOB_STATUSINFO_FILE))
        except FileNotFoundError: pass
        try:
            return JobStatusInfo.load(Path(DB_DIR_DELETED, job_id, JOB_STATUSINFO_FILE))
        except FileNotFoundError: pass
        return JobStatusInfo(id=job_id, status=JobStatus.NONEXISTENT)


# def process_job_fake(job_id: str, info: dict, duration: int):
#     time.sleep(5) #debug
#     job = rq.get_current_job()
#     qid = job.get_id()
#     Path(DB_DIR_RUNNING, job_id, JOB_STATUS_FILE).write_text('PROCESSING')
#     with Path(DB_DIR_RUNNING, job_id, JOB_RESULT_FILE).open('w') as w:
#         for k, v in info.items():
#             w.write(f'{k}: {v}\n\n')
#         w.write(f'Enqueued as {qid}\n\n')
#         w.flush()
#         for i in range(1, duration+1):
#             time.sleep(1)
#             w.write(f'{i}\n')
#             w.flush()
#         domains = Path(DB_DIR_RUNNING, job_id, JOB_DOMAINS_FILE).read_text()
#         w.write(domains + '\n\n')    
#         w.write('Completed\n')
#     Path(DB_DIR_RUNNING, job_id, JOB_STATUS_FILE).write_text('COMPLETED')
        