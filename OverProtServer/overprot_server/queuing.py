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

from .constants import DB_DIR_PENDING, DB_DIR_RUNNING, DB_DIR_COMPLETED, DB_DIR_FAILED, DB_DIR_ARCHIVED, DB_DIR_DELETED, JOB_STATUSINFO_FILE, JOB_INFO_FILE, JOB_DOMAINS_FILE, JOB_DATADIR, JOB_DATAZIP, JOB_RESULT_FILE, JOB_STDOUT_FILE, JOB_STDERR_FILE, JOB_ERROR_MESSAGE_FILE, OVERPROT_PYTHON, OVERPROT_PY, OVERPROT_STRUCTURE_SOURCE, TIME_FORMAT, COMPLETED_JOB_STORING_DAYS, QUEUE_NAME, JOB_TIMEOUT, JOB_CLEANUP_TIMEOUT
from .domain_parsing import Domain


def enqueue_job(job_name: str, domains: List[Domain], info: dict) -> Tuple['JobStatusInfo', Any]:
    job_status = JobStatusInfo(name=job_name)  # default status Pending
    Path(DB_DIR_PENDING, job_status.id).mkdir(parents=True, exist_ok=True)
    job_status.save(Path(DB_DIR_PENDING, job_status.id, JOB_STATUSINFO_FILE))
    Path(DB_DIR_PENDING, job_status.id, JOB_DOMAINS_FILE).write_text('\n'.join(str(domain) for domain in sorted(domains)))
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

    stdout_file = Path(DB_DIR_RUNNING, job_id, JOB_STDOUT_FILE)
    stderr_file = Path(DB_DIR_RUNNING, job_id, JOB_STDERR_FILE)
    error_message_file = Path(DB_DIR_RUNNING, job_id, JOB_ERROR_MESSAGE_FILE)
    result_file = Path(DB_DIR_RUNNING, job_id, JOB_RESULT_FILE)

    info = json.loads(Path(DB_DIR_RUNNING, job_id, JOB_INFO_FILE).read_text())
    with open(result_file, 'w') as w:
        for k, v in info.items():
            w.write(f'{k}: {v}\n\n')
        w.write('Running...\n\n')

    # time.sleep(10)
    with open(stdout_file, 'w') as w_out:
        with open(stderr_file, 'w') as w_err:
            try:
                proc = subprocess.run([OVERPROT_PYTHON, OVERPROT_PY, 
                                    job_id, Path(DB_DIR_RUNNING, job_id, JOB_DATADIR), 
                                    '--domains', Path(DB_DIR_RUNNING, job_id, JOB_DOMAINS_FILE),
                                    '--structure_source', OVERPROT_STRUCTURE_SOURCE], 
                                    check=True, timeout=JOB_TIMEOUT, stdout=w_out, stderr=w_err)
                proc_returncode: Optional[int] = proc.returncode
                successfull = True
                error_message = None
            except subprocess.CalledProcessError as ex:
                proc_returncode = ex.returncode
                successfull = False
                error_message = 'Please contact us and include the job ID in your message.'
                print(f'Unexpected error:\n{type(ex).__name__}: {ex}', file=w_err)
            except subprocess.TimeoutExpired as ex:
                proc_returncode = None
                successfull = False
                error_message = f'Job timeout expired ({JOB_TIMEOUT} seconds).'
                print(error_message, file=w_err)
            except Exception as ex:
                proc_returncode = None
                successfull = False
                error_message = 'Please contact us and include the job ID in your message.'
                print(f'Unexpected error:\n{type(ex).__name__}: {ex}', file=w_err)
            if error_message is not None:
                error_message_file.write_text(error_message)


    with open(result_file, 'a') as w:
        if successfull:
            w.write('Completed\n\n')
        elif proc_returncode is not None: 
            w.write(f'Failed (exit code: {proc_returncode})\n\n')
        else: 
            w.write(f'Failed (timed out)\n\n')

    # if not successfull and proc_returncode is not None:
    #     proc_stderr = Path(DB_DIR_RUNNING, job_id, JOB_STDERR_FILE).read_text()
    #     err_msg = extract_error_message(proc_stderr)
    #     error_message_file.write_text(err_msg)
        


    if successfull:
        # TODO select only important results
        shutil.copy(Path(DB_DIR_RUNNING, job_id, JOB_STDOUT_FILE), Path(DB_DIR_RUNNING, job_id, JOB_DATADIR, JOB_STDOUT_FILE))
        shutil.copy(Path(DB_DIR_RUNNING, job_id, JOB_STDERR_FILE), Path(DB_DIR_RUNNING, job_id, JOB_DATADIR, JOB_STDERR_FILE))
        make_archive(Path(DB_DIR_RUNNING, job_id, JOB_DATADIR), Path(DB_DIR_RUNNING, job_id, JOB_DATAZIP))
        # shutil.make_archive(Path(DB_DIR_RUNNING, job_id, JOB_DATAZIP), 'zip', Path(DB_DIR_RUNNING, job_id, JOB_DATADIR))
        # move(Path(DB_DIR_RUNNING, job_id, JOB_DATAZIP+'.zip'), Path(DB_DIR_RUNNING, job_id, JOB_DATAZIP))
        move(Path(DB_DIR_RUNNING, job_id, JOB_DATADIR, 'results'), Path(DB_DIR_RUNNING, job_id, 'results'))
        make_archive(Path(DB_DIR_RUNNING, job_id, 'results'), Path(DB_DIR_RUNNING, job_id, 'results.zip'))
        move(Path(DB_DIR_RUNNING, job_id, JOB_DATADIR, 'lists'), Path(DB_DIR_RUNNING, job_id, 'lists'))
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
