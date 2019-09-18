from flask import render_template, flash, redirect, url_for, request, session
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.urls import url_parse
from app import app, db
from app.forms import LoginForm, RegistrationForm, EditProfileForm, AnswerForm, GoldMSIForm
from app.models import User, Question
from datetime import datetime
from sqlalchemy import func
from wtforms import Label



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None:
            flash('Unknown username!')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        return redirect('index')
    return render_template('login.html', title='Returning participant', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        flash('To create a new profile, please log out.')
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for('index'))
    return render_template('register.html', title='Register', form=form)


@app.route('/user/<username>',methods=['GET','POST'])
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()

    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('user',username=current_user.username))
    elif request.method == 'GET':
        form.username.data = current_user.username

    return render_template('user.html', user=user, form=form)



@app.route('/instructions')
def instructions():
    session['question_id'] = current_user.next_question()
    return render_template('instructions.html')

@app.route('/question',methods=['GET','POST'])
@login_required
def question():
    n_question = current_user.number_answers()
    form = AnswerForm()
    question_id = session['question_id']
    current_question = Question.query.get(question_id)

    if form.validate_on_submit():
        # flash('Your answered:'+str(form.choice.data)+' to question :'+str(current_question.id)+' known '+str(form.known.data))
        answer = current_question.answer(form.choice.data,current_user,recognised=form.known.data)
        db.session.add(answer)
        db.session.commit()
        session['question_id'] = current_user.next_question()
        return redirect(url_for('question'))

    # flash(str([current_question.id, question_id]))
    return render_template('question.html', number=n_question+1,filepaths = current_question.get_filepaths(), form=form)

@app.route('/music_background',methods=['GET','POST'])
@login_required
def music_background():

    questions = [
        "I have never been complimented for my talents as a musical performer.",
        "I would not consider myself a musician.",
        "I engaged in regular, daily practice of a musical instrument (including voice) for ___ years. ",
        "At the peak of my interest, I practiced ___ hours per day on my primary instrument.",
        "I have had formal training in music theory for __ years",
        "I have had __ years of formal training on a musical instrument (including voice) during my lifetime. ",
        "I can play ___ musical instruments",
        ]
    choice_labels = [
        "Agreement Scale",
        "Agreement Scale",
        "0;1;2;3;4-5;6-9;10 or more",
        "0;0.5;1;1.5;2;3-4;5 or more",
        "0;0.5;1;2;3;4-6;7 or more",
        "0;0.5;1;2;3-5;6-9;10 or more",
        "0;1;2;3;4;5;6 or more",
        ]

    choice_labels = [['Completely disagree','Strongly disagree','Disagree','Neither agree or disagree','Agree','Strongly agree','Completely agree'] if elt == 'Agreement Scale' else elt.split(';')  for elt in choice_labels]


    form = GoldMSIForm()

    if len(form.all_choices.entries) ==0:
        if current_user.gold_msi_completed:
            answers = current_user.gold_msi_answers.split(';')
            for answer in answers:
                form.all_choices.append_entry({'choice':answer})
        else:
            for i in range(len(questions)):
                form.all_choices.append_entry()

    if form.validate_on_submit():
        flash('You completed the test!')
        current_user.gold_msi_answers = ';'.join([str(entry.data[u'choice']) for entry in form.all_choices.entries])
        current_user.gold_msi_completed = True
        db.session.commit()

        return redirect(url_for('instructions'))


    return render_template('music_background.html',form=form,questions_labels=zip(questions,choice_labels))


@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()
