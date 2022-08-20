import os
from pathlib import Path

try:
    OVERPROT_PYTHON = Path(os.environ['OVERPROT_PYTHON'])
    OVERPROT_PY = Path(os.environ['OVERPROT_PY'])
    VAR_DIR = Path(os.environ['VAR_DIR'])
    QUEUE_NAME = os.environ['RQ_QUEUE']
except KeyError as ex:
    raise Exception(f'Environment variable {ex} must be defined before running this program.')

DATA_DIR = Path(os.environ.get('DATA_DIR', VAR_DIR/'data'))  # Directory with freely available static data (route /data maps to this dir, but in deployment it will be served by nginx)
OVERPROT_STRUCTURE_SOURCE = os.environ.get('OVERPROT_STRUCTURE_SOURCE', '')

MAXIMUM_JOB_DOMAINS = int(os.environ.get('MAXIMUM_JOB_DOMAINS', 500))
JOB_TIMEOUT = int(os.environ.get('JOB_TIMEOUT', 86400))       # seconds, (86400 s = 24 hours)
JOB_CLEANUP_TIMEOUT = 600  # seconds, extra time for cleanup in case that job timed out
COMPLETED_JOB_STORING_DAYS = 14  # Currently not implemented


LAST_UPDATE_FILE = DATA_DIR/'db'/'last_update.txt'
JOBS_DIR = VAR_DIR/'jobs'
JOBS_DIR_PENDING = JOBS_DIR/'Pending'
JOBS_DIR_RUNNING = JOBS_DIR/'Running'
JOBS_DIR_COMPLETED = JOBS_DIR/'Completed'
JOBS_DIR_FAILED = JOBS_DIR/'Failed'
JOBS_DIR_ARCHIVED = JOBS_DIR/'Archived'
JOBS_DIR_DELETED = JOBS_DIR/'Deleted'

JOB_STATUSINFO_FILE = 'job_status.txt'
JOB_INFO_FILE = 'job_info.json'
JOB_DOMAINS_FILE = 'job_domain_list.txt'
JOB_DATADIR = 'job_data'
JOB_DATAZIP = 'data.zip'
JOB_RESULT_FILE = 'result.txt'
JOB_STDOUT_FILE = 'stdout.txt'
JOB_STDERR_FILE = 'stderr.txt'
JOB_ERROR_MESSAGE_FILE = 'error_message.txt'

DOMAIN_INFO_FILE_TEMPLATE = f'{DATA_DIR}/db/domain/info/{{domain_middle}}/{{domain}}.json'


REFRESH_TIMES = [2, 5, 10, 20, 30, 60, 120, 180, 300, 600, 1200, 1800, 3600]  # Times (in seconds) since submission when the waiting page should be autorefreshed, then each k*REFRESH_TIMES[-1] for any natural k 

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

DEFAULT_FAMILY_EXAMPLE = '1.10.630.10'  # Cytochrome P450
DEFAULT_DOMAIN_EXAMPLE = '1tqnA00'  # Cytochrome P450