from flask import Flask
from config import Config, DATA_PATH
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from logging.handlers import RotatingFileHandler
import logging
import os






app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login = LoginManager(app)
login.login_view = 'login'
bootstrap = Bootstrap(app)
moment = Moment(app)

app.jinja_env.filters['zip'] = zip


if not app.debug:

    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/website_AMT.log', maxBytes=10240,
                                       backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)

    app.logger.setLevel(logging.INFO)
    app.logger.info('Website AMT perception startup')

from app import routes, models, errors

db.create_all()
if models.Question.query.first() is None:
    print('Populate database...')

    data_path = os.path.join('app/static',DATA_PATH)
    all_folders = [path for path in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,path))]
    for folder in all_folders:
        example = folder
        folder_path = os.path.join(data_path,folder)
        files = [elt for elt in os.listdir(folder_path) if elt.endswith('.mp3') and not elt.startswith('.') and not 'target' in elt]
        n_files = len(files)
        for i in range(n_files):
            for j in range(i+1,n_files):
                system1,system2 = sorted([os.path.splitext(files[i])[0],os.path.splitext(files[j])[0]])
                question = models.Question(example=example,system1=os.path.splitext(files[i])[0],system2=os.path.splitext(files[j])[0])
                db.session.add(question)

    db.session.commit()
