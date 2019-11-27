import os
import datetime

basedir = os.path.abspath(os.path.dirname(__file__))


MAX_ANSWERS = 4
DATA_PATH = 'data/all_mp3_cut'
MIN_DATE = datetime.datetime(datetime.MINYEAR,1,1,0,0)
LOCK_TIME = datetime.timedelta(minutes=2) #Amount of timeor which other users cannot access the same question.


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'lol_tro_maran'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
