import os

try:
    OVERPROT_PYTHON = os.environ['OVERPROT_PYTHON']
    OVERPROT_PY = os.environ['OVERPROT_PY']
    VAR_DIR = os.environ['VAR_DIR']
    QUEUE_NAME = os.environ['RQ_QUEUE']
except KeyError as ex:
    raise Exception(f'Environment variable {ex} must be defined before running this program.')

DATA_DIR = os.environ.get('DATA_DIR', f'{VAR_DIR}/data')  # Directory with freely available static data (route /data maps to this dir, but in deployment it will be served by nginx)
OVERPROT_STRUCTURE_SOURCE = os.environ.get('OVERPROT_STRUCTURE_SOURCE', '')

MAXIMUM_JOB_DOMAINS = int(os.environ.get('MAXIMUM_JOB_DOMAINS', 500))
JOB_TIMEOUT = int(os.environ.get('JOB_TIMEOUT', 86400))       # seconds, (86400 s = 24 hours)
JOB_CLEANUP_TIMEOUT = 600  # seconds, extra time for cleanup in case that job timed out
COMPLETED_JOB_STORING_DAYS = 14  # Currently not implemented


LAST_UPDATE_FILE = f'{DATA_DIR}/db/LAST_UPDATE.txt'
DB_DIR = f'{VAR_DIR}/jobs'
DB_DIR_PENDING = f'{DB_DIR}/Pending'
DB_DIR_RUNNING = f'{DB_DIR}/Running'
DB_DIR_COMPLETED = f'{DB_DIR}/Completed'
DB_DIR_FAILED = f'{DB_DIR}/Failed'
DB_DIR_ARCHIVED = f'{DB_DIR}/Archived'
DB_DIR_DELETED = f'{DB_DIR}/Deleted'

JOB_STATUSINFO_FILE = 'job_status.txt'
JOB_INFO_FILE = 'job_info.json'
JOB_DOMAINS_FILE = 'job_domain_list.txt'
JOB_DATADIR = 'job_data'
JOB_DATAZIP = 'data.zip'
JOB_RESULT_FILE = 'result.txt'
JOB_STDOUT_FILE = 'stdout.txt'
JOB_STDERR_FILE = 'stderr.txt'
JOB_ERROR_MESSAGE_FILE = 'error_message.txt'


REFRESH_TIMES = [2, 5, 10, 20, 30, 60, 120, 180, 300, 600, 1200, 1800, 3600]  # Times (in seconds) since submission when the waiting page should be autorefreshed, then each k*REFRESH_TIMES[-1] for any natural k 

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

DEFAULT_FAMILY_EXAMPLE = '1.10.630.10'  # Cytochrome P450
DEFAULT_DOMAIN_EXAMPLE = '1tqnA00'  # Cytochrome P450