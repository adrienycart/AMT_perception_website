from flask import render_template
from app import app

@app.route('/')
@app.route('/index')
def index():
    question = {'number':1,'example':'lol',
                'systems':['haha','d_bar']}

    return render_template('index.html', question=question)
