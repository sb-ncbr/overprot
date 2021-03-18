import flask
from markupsafe import escape
from typing import Any, NamedTuple, Optional, Union


app = flask.Flask(__name__)


class ResponseTuple(NamedTuple):
    response: Union[str, dict]
    status: Optional[int] = None
    headers: Union[list, dict, None] = None


@app.route('/')
def index() -> Any:
    # flask.abort(401)
    return flask.redirect(flask.url_for('hello', name='Pica'))
    # return 'Index page'

@app.route('/hello/<name>')
def hello(name: str) -> str:
    name = escape(name.title())
    query = flask.request.args
    repeat = query.get('repeat', type=int, default=1)
    response = '<br>'.join(f'Hello {name}! {query}' for _ in range(repeat))
    return response

@app.route('/foo')
def foo() -> Any:
    response = {'foo': {'bar': 5, 'baz': tuple(range(5))}}
    return ResponseTuple(response, 202)
