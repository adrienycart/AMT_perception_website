import os

basedir = os.path.abspath(os.path.dirname(__file__))


MAX_ANSWERS = 3
DATA_PATH = '../data/all_mp3_cut'


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'lol_tro_maran'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
