from flask import Flask
from config import Config, DATA_PATH
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
from logging.handlers import RotatingFileHandler
import os






app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login = LoginManager(app)
login.login_view = 'login'
bootstrap = Bootstrap(app)

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
        for system1 in os.listdir(folder_path):
            if system1.endswith('.mp3') and not system1.startswith('.') and not 'target' in system1:
                for system2 in os.listdir(folder_path):
                    if system2.endswith('.mp3') and not system2.startswith('.') and not 'target' in system2 and not system1==system2:
                        question = models.Question(example=example,system1=os.path.splitext(system1)[0],system2=os.path.splitext(system2)[0])
                        db.session.add(question)
    db.session.commit()
