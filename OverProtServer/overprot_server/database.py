import sqlite3
import flask

def get_db():
    if 'db' not in flask.g:
        flask.g.db = _connect_to_db()
    return flask.g.db
   
def close_db(e=None):
    db = flask.g.pop('db', None)
    if db is not None:
        db.close()
        
def _connect_to_db():
    db = sqlite3.connect(flask.current_app.config['DATABASE'], detect_types=sqlite3.PARSE_DECLTYPES)
    db.row_factory = sqlite3.Row
    return db
 