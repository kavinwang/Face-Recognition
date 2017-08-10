import pymysql


def open_connection():
    config = {
        'host': '10.79.169.17',
        'port': 3306,
        'user': 'csairmind',
        'password': 'csairmind',
        'db': 'csair_face_recognition',
        'charset': 'utf8',
        'autocommit': True
    }
    conn = pymysql.connect(**config)
    return conn


def insert_model_info(conn, network, version, params):
    assert network in ['facenet', 'classifier'], 'network must be facenet or classifier'
    if network == 'facenet':
        sql = 'insert into model_info values (%s, %s)'
    elif network == 'classifier':
        sql = 'insert into classifier_info values (%s, %s)'
    else:
        raise ValueError
    cur = conn.cursor()
    cur.executemany(sql, [(str(version), params)])
    cur.close()


def get_model_info(conn, network, version):
    assert network in ['facenet', 'classifier', 'candidate'], 'network must be facenet or classifier'
    if network == 'facenet':
        sql = 'select parameters from model_info where version=(%s)'
    else:
        sql = 'select parameters from classifier_info where version=(%s)'
    cur = conn.cursor()
    cur.execute(sql, version)
    info = cur.fetchone()
    cur.close()
    return info


def clean_model_info(conn, network, version):
    assert network in ['facenet', 'classifier', 'candidate'], 'network must be facenet or classifier or candidate'
    if network == 'facenet':
        sql = 'delete from model_info where version=(%s)'
    elif network == 'classifier':
        sql = 'delete from classifier_info where version=(%s)'
    elif network == 'candidate':
        sql = 'delete from candidate_info'
    else:
        raise ValueError
    cur = conn.cursor()
    cur.execute(sql, version)
    cur.close()


def insert_candidate_info(conn, params):
    cur = conn.cursor()
    sql = 'insert into candidate_info values (%s, %s)'
    cur.executemany(sql, params)
    cur.close()


def get_candidate_info(conn, num):
    cur = conn.cursor()
    sql = 'select name from candidate_info where num = %s'
    cur.execute(sql, num)
    name = cur.fetchone()
    cur.close()
    return name


def close_connection(conn):
    conn.close()
