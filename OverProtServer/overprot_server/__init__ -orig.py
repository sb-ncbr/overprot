import flask
from markupsafe import escape
import uuid
from datetime import timedelta
from http import HTTPStatus
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union, Dict, Set

from . import constants
from .constants import DATA_DIR, JOBS_DIR_PENDING, JOBS_DIR_RUNNING, JOBS_DIR_COMPLETED, JOBS_DIR_ARCHIVED, JOBS_DIR_FAILED, JOB_ERROR_MESSAGE_FILE, MAXIMUM_JOB_DOMAINS, REFRESH_TIMES, DEFAULT_FAMILY_EXAMPLE, DEFAULT_DOMAIN_EXAMPLE, LAST_UPDATE_FILE
from .data_caching import DataCache
from .searching import Searcher
from . import domain_parsing
from . import queuing


app = flask.Flask(__name__)
app.url_map.strict_slashes = False  # This is needed because some browsers (e.g. older Opera) sometimes add trailing /, where you don't want them (WTF?)

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
    return flask.redirect(flask.url_for('home'))

@app.route('/home')
def home() -> Any:
    return flask.render_template('home.html')

@app.route('/submit')
def submit() -> Any:
    values = flask.request.values  # flask.request.args from GET + flask.request.form from POST
    job_name = values.get('job_name', type=str) or ''
    domain_list = (values.get('domain_list', type=str) or '').replace(';', '\n') 
    return flask.render_template('submit.html', job_name=job_name, domain_list=domain_list)


@app.route('/submission', methods=['POST'])
def submission_post() -> Any:
    values = flask.request.values  # flask.request.args from GET + flask.request.form from POST
    job_name: str = values.get('job_name', type=str) or ''
    numbering: str = values.get('numbering', type=str) or 'label'
    if numbering != 'label':
        return ResponseTuple('Not implemented: Residue numbering scheme "auth" is not implemented yet. Please use the "label" scheme.', HTTPStatus.NOT_IMPLEMENTED)
    list_text: str = (values.get('list', type=str) or '').strip()
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
        refresh_time = _calculate_time_to_refresh(elapsed_time)
        url = flask.request.url
        return flask.render_template('waiting.html', job_id=job_id, job_name=job_status.name, job_status=job_status.status.value, 
                                     submission_time=job_status.submission_time, current_time=now, elapsed_time=elapsed_time, 
                                     url=url, refresh_time=refresh_time)
    elif job_status.status in [queuing.JobStatus.COMPLETED, queuing.JobStatus.ARCHIVED]:
        file = flask.url_for('diagram', job_id=job_id)
        fam_info = _family_info_for_job(job_id)
        return flask.render_template('completed.html', file=file, job_id=job_id, job_name=job_status.name, job_status=job_status.status.value, 
                                    submission_time=job_status.submission_time, family_info=fam_info)
    elif job_status.status == queuing.JobStatus.FAILED:
        try:
            error_message = Path(JOBS_DIR_FAILED, job_id, JOB_ERROR_MESSAGE_FILE).read_text()
        except FileNotFoundError:
            error_message = ''
        return flask.render_template('failed.html', job_id=job_id, job_name=job_status.name, job_status=job_status.status.value, 
                                    submission_time=job_status.submission_time, error_message=error_message)
    elif job_status.status == queuing.JobStatus.DELETED:
        return ResponseTuple.plain(f'Job {job_id} expired ({job_status.delete_time}) and has been deleted.', status=HTTPStatus.NOT_FOUND)
    else:
        raise AssertionError(f'Unknow job status: {job_status.status}')


@app.route('/search', methods=['GET'])
def search() -> Any:
    values = flask.request.values 
    query: str = values.get('q', '')
    query = query.strip()

    if _family_exists(query):
        return flask.redirect(flask.url_for('family_view', family_id=query))

    pdb = SEARCHER_CACHE.value.search_pdb(query)
    if pdb is not None:
        return flask.redirect(flask.url_for('pdb', pdb_id=pdb))
    
    domain = SEARCHER_CACHE.value.search_domain(query)
    if domain is not None:
        return flask.redirect(flask.url_for('domain', domain_id=domain))
    
    return flask.render_template('search_fail.html', entity=query)

@app.route('/pdb/<string:pdb_id>', methods=['GET'])
def pdb(pdb_id: str) -> Any:
    if SEARCHER_CACHE.value.has_pdb(pdb_id):
        domains_families = SEARCHER_CACHE.value.get_domains_families_for_pdb(pdb_id)
        return flask.render_template('pdb.html', pdb=pdb_id, n_domains=len(domains_families), domains_families=domains_families)
    else:
        return flask.render_template('404.html', entity_type='PDB entry', entity_id=pdb_id), HTTPStatus.NOT_FOUND

@app.route('/domain/<string:domain_id>', methods=['GET'])
def domain(domain_id: str) -> Any:
    if SEARCHER_CACHE.value.has_domain(domain_id):
        family_id = SEARCHER_CACHE.value.get_family_for_domain(domain_id)
        # return flask.redirect(flask.url_for('domain_view', family_id=family_id, domain_id=domain_id)) # TODO uncomment
        # DEBUG:
        pdb_id = SEARCHER_CACHE.value.get_pdb_for_domain(domain_id)
        chain, ranges = SEARCHER_CACHE.value.get_chain_ranges_for_domain(domain_id)
        return flask.render_template('domain.html', domain=domain_id, pdb=pdb_id, family=family_id, chain=chain, ranges=ranges)
    else:
        return flask.render_template('404.html', entity_type='Domain', entity_id=domain_id), HTTPStatus.NOT_FOUND

@app.route('/test/<string:domain_id>', methods=['GET'])
def test(domain_id: str) -> Any:
    domain = SEARCHER_CACHE.value.search_domain(domain_id)
    return {
        'PDBs': len(SEARCHER_CACHE.value._pdbs),
        'Domains': len(SEARCHER_CACHE.value._domains),
        'Families': len(SEARCHER_CACHE.value._families),
        'Domain': repr(domain),
    }

@app.route('/family_view', methods=['GET'])
def family_view() -> Any:
    values = flask.request.values 
    family_id = values.get('family_id', None)
    if family_id is None:
        return flask.redirect(flask.url_for('family_view', family_id=DEFAULT_FAMILY_EXAMPLE, example=1))
    else:
        if _family_exists(family_id):
            fam_info = _family_info(family_id)
            example_domain = EXAMPLE_DOMAINS_CACHE.value.get(family_id)
            return flask.render_template('family_view.html', family_id=family_id, family_info=fam_info, example_domain=example_domain, last_update=LAST_UPDATE_CACHE.value)
        else:
            return flask.render_template('404.html', entity_type='Family', entity_id=family_id), HTTPStatus.NOT_FOUND

@app.route('/domain_view', methods=['GET'])
def domain_view() -> Any:
    values = flask.request.values 
    family_id = values.get('family_id', None)
    domain_id = values.get('domain_id', None)
    if domain_id is not None and SEARCHER_CACHE.value.has_domain(domain_id):
        pdb_id = SEARCHER_CACHE.value.get_pdb_for_domain(domain_id)
        family_id = SEARCHER_CACHE.value.get_family_for_domain(domain_id)
        chain, ranges = SEARCHER_CACHE.value.get_chain_ranges_for_domain(domain_id)
        # return flask.redirect(f'/static/integration/index.html?family_id={family_id}&domain_id={domain_id}')  # debug
        if family_id is None:
            return flask.redirect(flask.url_for('domain_view', family_id=DEFAULT_FAMILY_EXAMPLE, domain_id=DEFAULT_DOMAIN_EXAMPLE, example=1))
        else:
            if _family_exists(family_id):
                fam_info = _family_info(family_id)
                return flask.render_template('domain_view.html', family_id=family_id, domain_id=domain_id, pdb=pdb_id, chain=chain, ranges=ranges)
            else:
                return flask.render_template('search_fail.html', entity=family_id)
    else:
        return flask.render_template('404.html', entity_type='Domain', entity_id=domain_id), HTTPStatus.NOT_FOUND

@app.route('/favicon.ico')
def favicon() -> Any:
    return flask.send_file('static/images/favicon.ico')


@app.route('/api_doc')
def api_doc() -> Any:
    url_root = flask.request.url_root.rstrip('/')
    return flask.render_template('api_doc.html', root=url_root)

@app.route('/api/domain/<string:endpoint>/<string:file>')
def domain_api(endpoint: str, file: str) -> Any:
    '''file should start with domain_id'''
    subdir = file[1:3]
    return flask.redirect(f'/data/db/domain/{endpoint}/{subdir}/{file}')

@app.route('/base')  # debug
def base() -> Any:
    return ResponseTuple.plain(flask.request.url_root)

@app.route('/diagram/<string:job_id>')
def diagram(job_id: str) -> Any:
    try:
        return flask.send_file(Path(JOBS_DIR_COMPLETED, job_id, 'results', 'diagram.json'))
    except FileNotFoundError:
        return flask.send_file(Path(JOBS_DIR_ARCHIVED, job_id, 'results', 'diagram.json'))

@app.route('/results/<string:job_id>/<path:file>')
def results(job_id: str, file: str) -> Any:
    if '..' in file:
        return ResponseTuple(f'{file} not available', status=HTTPStatus.FORBIDDEN)
    try:
        return flask.send_file(Path(JOBS_DIR_COMPLETED, job_id, file))
    except FileNotFoundError:
        try:
            return flask.send_file(Path(JOBS_DIR_ARCHIVED, job_id, file))
        except FileNotFoundError:
            return flask.send_file(Path(JOBS_DIR_FAILED, job_id, file))

# DEBUG
@app.route('/settings')
def settings() -> Any:
    pairs = {
        'OVERPROT_PYTHON': constants.OVERPROT_PYTHON,
        'OVERPROT_PY': constants.OVERPROT_PY,
        'OVERPROT_STRUCTURE_SOURCE': constants.OVERPROT_STRUCTURE_SOURCE,
        'VAR_DIR': constants.VAR_DIR,
        'DATA_DIR': constants.DATA_DIR,
        'QUEUE_NAME': constants.QUEUE_NAME,
        'MAXIMUM_JOB_DOMAINS': constants.MAXIMUM_JOB_DOMAINS,
        'JOB_TIMEOUT': constants.JOB_TIMEOUT,
        'JOB_CLEANUP_TIMEOUT': constants.JOB_CLEANUP_TIMEOUT,
        'COMPLETED_JOB_STORING_DAYS': constants.COMPLETED_JOB_STORING_DAYS,
    }
    key_width = max(len(key) for key in pairs.keys())
    text = '\n'.join(f'{k:{key_width}}: {v}' for k, v in pairs.items())
    return ResponseTuple.plain(text)

# DEBUG
@app.route('/data_cache')
def data_cache() -> Any:
    vu1 = EXAMPLE_DOMAINS_CACHE._valid_until
    value = EXAMPLE_DOMAINS_CACHE.value
    vu2 = EXAMPLE_DOMAINS_CACHE._valid_until
    text = '\n'.join(str(x) for x in [vu1, vu2, '', value])
    return ResponseTuple.plain(text)

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


def _calculate_time_to_refresh(elapsed_time: timedelta) -> int:
    elapsed_seconds = int(elapsed_time.total_seconds())
    try:
        next_refresh = next(t for t in REFRESH_TIMES if t > elapsed_seconds)
        return next_refresh - elapsed_seconds
    except StopIteration:
        return -elapsed_seconds % REFRESH_TIMES[-1]

def _family_exists(family_id: str) -> bool:
    # return SEARCHER_CACHE.value.has_family(family_id)
    return family_id in FAMILY_SET_CACHE.value
    # return Path(DATA_DIR, 'db', 'diagrams', f'diagram-{family_id}.json').exists()

def _family_info(family_id: str) -> Dict[str, str]:
    SEP = ':'
    result = {}
    try:
        with open(Path(DATA_DIR, 'db', 'family', 'lists', family_id, f'family_info.txt')) as f:
            for line in f:
                if SEP in line:
                    key, value = line.split(SEP, maxsplit=1)
                    result[key.strip()] = value.strip()
        return result
    except OSError:
        return {}

def _family_info_for_job(job_id: str) -> Dict[str, str]:
    SEP = ':'
    result = {}
    try:
        family_info_file = _get_job_file(job_id, 'lists', 'family_info.txt')
    except FileNotFoundError:
        return {}
    with open(family_info_file) as f:
        for line in f:
            if SEP in line:
                key, value = line.split(SEP, maxsplit=1)
                result[key.strip()] = value.strip()
    return result

def _get_job_file(job_id: str, *path_parts: str) -> Path:
    '''Get path to file jobs/*/job_id/*path_parts of given job, where * can be Completed, Archived, Failed, or Pending etc. '''
    for db_dir in (JOBS_DIR_COMPLETED, JOBS_DIR_ARCHIVED, JOBS_DIR_FAILED, JOBS_DIR_PENDING):
        path = Path(db_dir, job_id, *path_parts)
        if path.exists():
            return path
    raise FileNotFoundError('/'.join(['...', job_id, *path_parts]))

def _get_example_domains() -> Dict[str, str]:
    SEPARATOR = ';'
    with open(Path(DATA_DIR, 'db', 'cath_example_domains.csv')) as f:
        result = {}
        for line in f:
            line = line.strip()
            family, example = line.split(SEPARATOR)
            result[family] = example
    return result

def _get_family_set() -> Set[str]:
    with open(Path(DATA_DIR, 'db', 'families.txt')) as f:
        result = set()
        for line in f:
            family_id = line.strip()
            if family_id != '':
                result.add(family_id)
    return result

def _get_last_update() -> str:
    try:
        return Path(LAST_UPDATE_FILE).read_text().strip()
    except OSError:
        return '???'
 

EXAMPLE_DOMAINS_CACHE = DataCache(_get_example_domains)
SEARCHER_CACHE = DataCache(lambda: Searcher(Path(DATA_DIR, 'db', 'domain_list.csv'), pdb_list_txt=Path(DATA_DIR, 'db', 'pdbs.txt')))
FAMILY_SET_CACHE = DataCache(_get_family_set)
LAST_UPDATE_CACHE = DataCache(_get_last_update)
