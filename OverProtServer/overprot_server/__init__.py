import flask
from markupsafe import escape
from http import HTTPStatus
from pathlib import Path
from typing import Any

from . import constants
from .constants import DATA_DIR, JOBS_DIR_COMPLETED, JOBS_DIR_ARCHIVED, JOBS_DIR_FAILED, JOB_ERROR_MESSAGE_FILE, MAXIMUM_JOB_DOMAINS, DEFAULT_FAMILY_EXAMPLE, DEFAULT_DOMAIN_EXAMPLE
from .search_results import SearchResults
from . import domain_parsing
from . import queuing
from .helpers import ResponseTuple, SEARCHER_CACHE, LAST_UPDATE_CACHE, EXAMPLE_DOMAINS_CACHE, calculate_time_to_refresh, family_exists, get_domain_info, get_family_info, get_family_info_for_job


app = flask.Flask(__name__)
app.url_map.strict_slashes = False  # This is needed because some browsers (e.g. older Opera) sometimes add trailing /, where you don't want them (WTF?)


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
        refresh_time = calculate_time_to_refresh(elapsed_time)
        url = flask.request.url
        return flask.render_template('waiting.html', job_id=job_id, job_name=job_status.name, job_status=job_status.status.value, 
                                     submission_time=job_status.submission_time, current_time=now, elapsed_time=elapsed_time, 
                                     url=url, refresh_time=refresh_time)
    elif job_status.status in [queuing.JobStatus.COMPLETED, queuing.JobStatus.ARCHIVED]:
        file = flask.url_for('diagram', job_id=job_id)
        fam_info = get_family_info_for_job(job_id)
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
    if family_exists(query):
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
    if not family_exists(family_id):
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
        if family_exists(family_id):
            fam_info = get_family_info(family_id)
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

