import flask
from markupsafe import escape
import uuid
from datetime import timedelta
from http import HTTPStatus
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union, Dict

from .constants import DATA_DIR, DB_DIR_RUNNING, DB_DIR_COMPLETED, DB_DIR_ARCHIVED, DB_DIR_FAILED, JOB_RESULT_FILE, JOB_ERROR_MESSAGE_FILE, MAXIMUM_JOB_DOMAINS, REFRESH_TIMES, DEFAULT_FAMILY_EXAMPLE, LAST_UPDATE_FILE
from . import domain_parsing
from . import queuing


app = flask.Flask(__name__)

class ResponseTuple(NamedTuple):
    response: Union[str, dict]
    status: Optional[int] = None
    headers: Union[list, dict, None] = None
    @classmethod
    def plain(cls, response, status=None, headers=None) -> 'ResponseTuple':
        if headers is None:
            headers = {'Content-Type': 'text/plain; charset=utf-8'}
        else:
            headers['Content-Type'] = 'text/plain; charset=utf-8'
        return cls(str(response), status, headers)


def get_uuid() -> uuid.UUID:
    return uuid.uuid4()


@app.route('/')
def index() -> Any:
    return flask.redirect('/home')

@app.route('/home')
def home() -> Any:
    return flask.render_template('home.html')

@app.route('/submit')
def submit() -> Any:
    values = flask.request.values  # flask.request.args from GET + flask.request.form from POST
    job_name = values.get('job_name', type=str, default='')
    domain_list = values.get('domain_list', type=str, default='').replace(';', '\n')
    return flask.render_template('submit.html', job_name=job_name, domain_list=domain_list)


@app.route('/submission', methods=['POST'])
def submission_post() -> Any:
    values = flask.request.values  # flask.request.args from GET + flask.request.form from POST
    job_name = values.get('job_name', type=str, default='')
    numbering = values.get('numbering', type=str, default='label')
    if numbering != 'label':
        return ResponseTuple('Not implemented: Residue numbering scheme "auth" is not implemented yet. Please use the "label" scheme.', HTTPStatus.NOT_IMPLEMENTED)
    list_text = values.get('list', type=str, default='').strip()
    list_file = flask.request.files['list_file']
    file_content = list_file.read().decode().strip()

    if list_text != '': 
        lst = list_text
    elif file_content != '':
        lst = file_content
    else:
        return ResponseTuple('Invalid request: empty list', HTTPStatus.BAD_REQUEST)

    try:
        domain_list = domain_parsing.parse_submission_list(lst)
    except domain_parsing.DomainParsingError as ex:
        return ResponseTuple(str(ex), HTTPStatus.BAD_REQUEST)

    if len(domain_list) > MAXIMUM_JOB_DOMAINS:
        return ResponseTuple(f'Error: The number of domains ({len(domain_list)}) exceeded the allowed limit ({MAXIMUM_JOB_DOMAINS}).', HTTPStatus.BAD_REQUEST)

    job_info = {'Job name': job_name, 'Numbering': numbering}
    job_status, qid = queuing.enqueue_job(job_name, domain_list, job_info)
    job_id = job_status.id

    return flask.redirect(flask.url_for('submitted', job_id=job_id))
    # return flask.redirect(flask.url_for('job', job_id=job_id))

@app.route('/submitted/<string:job_id>')
def submitted(job_id: str) -> Any:
    '''Show Submitted page (meaning that the job has been submitted successfully), which just sends GoogleAnalytics and redirects to /job/<job_id>.'''
    return flask.render_template('submitted.html', job_id=job_id)

@app.route('/job/<string:job_id>', methods=['GET'])
def job(job_id: str) -> Any:
    job_status = queuing.JobStatusInfo.load_for_job(job_id)
    if job_status.status == queuing.JobStatus.NONEXISTENT:
        return ResponseTuple.plain(f'Job {job_id} does not exist.', status=HTTPStatus.NOT_FOUND)
    elif job_status.status in [queuing.JobStatus.PENDING, queuing.JobStatus.RUNNING]:
        now = queuing.get_current_time()
        elapsed_time = now - job_status.submission_time
        refresh_time = calculate_time_to_refresh(elapsed_time)
        url = flask.request.url
        return flask.render_template('waiting.html', job_id=job_id, job_name=job_status.name, job_status=job_status.status.value, 
                                     submission_time=job_status.submission_time, current_time=now, elapsed_time=elapsed_time, 
                                     url=url, refresh_time=refresh_time)
    elif job_status.status in [queuing.JobStatus.COMPLETED, queuing.JobStatus.ARCHIVED]:
        file = flask.url_for('diagram', job_id=job_id)
        return flask.render_template('completed.html', file=file, job_id=job_id, job_name=job_status.name, job_status=job_status.status.value, 
                                    submission_time=job_status.submission_time)
    elif job_status.status == queuing.JobStatus.FAILED:
        try:
            error_message = Path(DB_DIR_FAILED, job_id, JOB_ERROR_MESSAGE_FILE).read_text()
        except FileNotFoundError:
            error_message = ''
        return flask.render_template('failed.html', job_id=job_id, job_name=job_status.name, job_status=job_status.status.value, 
                                    submission_time=job_status.submission_time, error_message=error_message)
    elif job_status.status == queuing.JobStatus.DELETED:
        return ResponseTuple.plain(f'Job {job_id} expired ({job_status.delete_time}) and has been deleted.', status=HTTPStatus.NOT_FOUND)
    else:
        raise AssertionError(f'Unknow job status: {job_status.status}')


@app.route('/view/', methods=['GET'])  # trailing / is needed because Opera automatically adds one (WTF?)
def view() -> Any:
    values = flask.request.values 
    family = values.get('family', None)
    if family is None:
        return flask.redirect(flask.url_for('view', family=DEFAULT_FAMILY_EXAMPLE, example=1))
    else:
        if family_exists(family):
            fam_info = family_info(family)
            return flask.render_template('view.html', family=family, family_info=fam_info, last_update=get_last_update())
        else:
            return flask.render_template('search_fail.html', family=family)

def get_last_update() -> str:
    try:
        return Path(LAST_UPDATE_FILE).read_text().strip()
    except OSError:
        return '???'
    

@app.route('/favicon.ico')
def favicon() -> Any:
    return flask.send_file('static/images/favicon.ico')

@app.route('/diagram/<string:job_id>')
def diagram(job_id: str) -> Any:
    try:
        return flask.send_file(Path(DB_DIR_COMPLETED, job_id, 'results', 'diagram.json'))
    except FileNotFoundError:
        return flask.send_file(Path(DB_DIR_ARCHIVED, job_id, 'results', 'diagram.json'))


@app.route('/results/<string:job_id>/<path:file>')
def results(job_id: str, file: str) -> Any:
    if '..' in file:
        return ResponseTuple(f'{file} not available', status=HTTPStatus.FORBIDDEN)
    try:
        return flask.send_file(Path(DB_DIR_COMPLETED, job_id, file))
    except FileNotFoundError:
        try:
            return flask.send_file(Path(DB_DIR_ARCHIVED, job_id, file))
        except FileNotFoundError:
            return flask.send_file(Path(DB_DIR_FAILED, job_id, file))


# DEBUG
@app.route('/data/<path:file>')
def data(file: str):
    if '..' in file:
        return ResponseTuple(f'{file} not available', status=HTTPStatus.FORBIDDEN)
    try:
        return flask.send_file(Path(DATA_DIR, file))
    except FileNotFoundError:
        return ResponseTuple(f'File {file} does not exist.', status=HTTPStatus.NOT_FOUND)
    # return f'{file} -- {escape(file)}'


def calculate_time_to_refresh(elapsed_time: timedelta) -> int:
    elapsed_seconds = int(elapsed_time.total_seconds())
    try:
        next_refresh = next(t for t in REFRESH_TIMES if t > elapsed_seconds)
        return next_refresh - elapsed_seconds
    except StopIteration:
        return -elapsed_seconds % REFRESH_TIMES[-1]

def family_exists(family_id: str) -> bool:
    return Path(DATA_DIR, 'db', 'diagrams', f'diagram-{family_id}.json').exists()

def family_info(family_id: str) -> Dict[str, str]:
    SEP = ':'
    result = {}
    try:
        with open(Path(DATA_DIR, 'db', 'families', family_id, f'family_info.txt')) as f:
            for line in f:
                if SEP in line:
                    key, value = line.split(SEP, maxsplit=1)
                    result[key.strip()] = value.strip()
        return result
    except OSError:
        return {}






