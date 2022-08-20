import flask
from markupsafe import escape
import uuid
from datetime import timedelta, datetime
from http import HTTPStatus
import json
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union, Dict, Set

from . import constants
from .constants import DATA_DIR, JOBS_DIR_PENDING, JOBS_DIR_RUNNING, JOBS_DIR_COMPLETED, JOBS_DIR_ARCHIVED, JOBS_DIR_FAILED, JOB_ERROR_MESSAGE_FILE, MAXIMUM_JOB_DOMAINS, REFRESH_TIMES, DEFAULT_FAMILY_EXAMPLE, DEFAULT_DOMAIN_EXAMPLE, LAST_UPDATE_FILE
from .data_caching import DataCache, DataCacheWithWatchfiles
from .searching import Searcher
from .search_results import SearchResult, SearchResults
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
    query = values.get('q') or ''
    query = query.strip()

    # Search families
    if _family_exists(query):
        return flask.redirect(flask.url_for('family_view', family_id=query, q=query))

    # Search PDBs
    pdbs = SEARCHER_CACHE.value.search_pdb(query)
    if len(pdbs) == 1:
        return flask.redirect(flask.url_for('pdb', pdb_id=pdbs[0]))
    elif len(pdbs) > 1:
        return flask.render_template('search_results.html', query=query, todo='TODO', results=SearchResults.from_pdbs(pdbs))
    
    # Search domains
    domains = SEARCHER_CACHE.value.search_domain(query)
    if len(domains) == 1:
        return flask.redirect(flask.url_for('domain', domain_id=domains[0]))
    elif len(domains) > 1:
        return flask.render_template('search_results.html', query=query, todo='TODO', results=SearchResults.from_domains(domains))

    # Search full-text - TODO

    return flask.render_template('search_results.html', query=query, todo='TODO', results=SearchResults())

    
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
        domain_info = get_domain_info(domain_id)
        return flask.redirect(flask.url_for('domain_view', family_id=domain_info.family, domain_id=domain_id))
        # return flask.render_template('domain.html', domain=domain_id, pdb=domain_info.pdb, family=domain_info.family, 
        #     chain=domain_info.chain_id, ranges=domain_info.ranges, 
        #     auth_chain= f'[auth {domain_info.auth_chain_id}]' if domain_info.chain_id != domain_info.auth_chain_id else '',
        #     auth_ranges = f'[auth {domain_info.auth_ranges}]' if domain_info.ranges != domain_info.auth_ranges else '')
    else:
        return flask.render_template('404.html', entity_type='Domain', entity_id=domain_id), HTTPStatus.NOT_FOUND
        
@app.route('/domain_view', methods=['GET'])
def domain_view() -> Any:
    values = flask.request.values 
    family_id = values.get('family_id')
    domain_id = values.get('domain_id')
    if family_id is None and domain_id is None:
        return flask.redirect(flask.url_for('domain_view', family_id=DEFAULT_FAMILY_EXAMPLE, domain_id=DEFAULT_DOMAIN_EXAMPLE, example=1))
    if family_id is None or domain_id is None:
        return ResponseTuple('Must either specify both family_id and domain_id, or none of them (redirects to example)', status=HTTPStatus.UNPROCESSABLE_ENTITY)
    if not _family_exists(family_id):
        return flask.render_template('404.html', entity_type='Family', entity_id=family_id), HTTPStatus.NOT_FOUND
    if not SEARCHER_CACHE.value.has_domain(domain_id):
        return flask.render_template('404.html', entity_type='Domain', entity_id=domain_id), HTTPStatus.NOT_FOUND
    domain_info = get_domain_info(domain_id)
    if family_id != domain_info.family:
        return ResponseTuple(f'Domain {domain_id} does not match family {family_id}', status=HTTPStatus.UNPROCESSABLE_ENTITY)
    return flask.render_template('domain_view.html', family_id=family_id, domain_id=domain_id, pdb=domain_info.pdb, 
        chain=domain_info.chain_id, ranges=domain_info.ranges, 
        auth_chain= f'[auth {domain_info.auth_chain_id}]' if domain_info.chain_id != domain_info.auth_chain_id else '',
        auth_ranges = f'[auth {domain_info.auth_ranges}]' if domain_info.ranges != domain_info.auth_ranges else '')
    # return flask.redirect(f'/static/integration/index.html?family_id={family_id}&domain_id={domain_id}')

@app.route('/test/<string:domain_id>', methods=['GET'])
def test(domain_id: str) -> Any:
    domains = SEARCHER_CACHE.value.search_domain(domain_id)
    return {'domains': domains}


@app.route('/family/<string:family_id>', methods=['GET'])
def family(family_id: str) -> Any:
    return flask.redirect(flask.url_for('family_view', family_id=family_id))
   
@app.route('/family_view', methods=['GET'])
def family_view() -> Any:
    values = flask.request.values 
    family_id = values.get('family_id')
    if family_id is None:
        return flask.redirect(flask.url_for('family_view', family_id=DEFAULT_FAMILY_EXAMPLE, example=1))
    else:
        if _family_exists(family_id):
            fam_info = _family_info(family_id)
            example_domain = EXAMPLE_DOMAINS_CACHE.value.get(family_id)
            return flask.render_template('family_view.html', family_id=family_id, family_info=fam_info, example_domain=example_domain, 
                last_update=LAST_UPDATE_CACHE.value, query=values.get('q', ''))
        else:
            return flask.render_template('404.html', entity_type='Family', entity_id=family_id), HTTPStatus.NOT_FOUND


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
    '''More efficient than using Searcher (reads less data)'''
    return family_id in FAMILY_SET_CACHE.value

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


DOMAIN_LIST_FILE = Path(DATA_DIR, 'db', 'domain_list.csv')
PDB_LIST_FILE = Path(DATA_DIR, 'db', 'pdbs.txt')
EXAMPLE_DOMAINS_FILE = Path(DATA_DIR, 'db', 'cath_example_domains.csv')
FAMILY_LIST_FILE = Path(DATA_DIR, 'db', 'families.txt')

def _get_example_domains() -> Dict[str, str]:
    SEPARATOR = ';'
    with open(EXAMPLE_DOMAINS_FILE) as f:
        result = {}
        for line in f:
            line = line.strip()
            family, example = line.split(SEPARATOR)
            result[family] = example
    return result

def _get_family_set() -> Set[str]:
    with open(FAMILY_LIST_FILE) as f:
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

class DomainInfo(NamedTuple):
    domain: str
    pdb: str
    chain_id: str
    ranges: str
    auth_chain_id: str
    auth_ranges: str
    family: str
    def format_chain(self) -> str:
        if self.chain_id == self.auth_chain_id:
            return self.chain_id
        else:
            return f'{self.chain_id} [auth {self.auth_chain_id}]'
    def format_ranges(self) -> str:
        if self.ranges == self.auth_ranges:
            return self.ranges
        else:
            return f'{self.ranges} [auth {self.auth_ranges}]'
    def format_auth_chain(self) -> str:
        if self.chain_id == self.auth_chain_id:
            return ''
        else:
            return f'[auth {self.auth_chain_id}]'
    def format_auth_ranges(self) -> str:
        if self.chain_id == self.auth_chain_id:
            return ''
        else:
            return f'[auth {self.auth_chain_id}]'

def get_domain_info(domain_id: str) -> DomainInfo:
    path = constants.DOMAIN_INFO_FILE_TEMPLATE.format(domain=domain_id, domain_middle=domain_id[1:3])
    try: 
        js = json.loads(Path(path).read_text())
        return DomainInfo(**js)
    except OSError:
        return DomainInfo(domain_id, '?', '?', '?', '?', '?', '?')

EXAMPLE_DOMAINS_CACHE = DataCacheWithWatchfiles(_get_example_domains, [EXAMPLE_DOMAINS_FILE])
SEARCHER_CACHE = DataCacheWithWatchfiles(lambda: Searcher(DOMAIN_LIST_FILE, pdb_list_txt=PDB_LIST_FILE), [DOMAIN_LIST_FILE, PDB_LIST_FILE])
FAMILY_SET_CACHE = DataCacheWithWatchfiles(_get_family_set, [FAMILY_LIST_FILE])
LAST_UPDATE_CACHE = DataCacheWithWatchfiles(_get_last_update, [LAST_UPDATE_FILE])
