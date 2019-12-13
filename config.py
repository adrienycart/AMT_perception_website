import os
import datetime
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

MAX_ANSWERS = 4
N_MODELS = 4
DATA_PATH = 'data/all_mp3_cut'
MIN_DATE = datetime.datetime(datetime.MINYEAR,1,1,0,0)
LOCK_TIME = datetime.timedelta(minutes=5) #Amount of time for which other users cannot access the same question.
BIAS_MOST_ANSWERS = 0.15 # Should be a value between 0 and 1. 1: No bias, 0: always choose most answered

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'lol_tro_maran'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
