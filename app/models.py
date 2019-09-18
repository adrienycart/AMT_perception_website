from app import db
from app import login
from datetime import datetime
from flask_login import UserMixin
import random
import os
from config import DATA_PATH
# from  sqlalchemy.sql.expression import func




class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    question_id = db.Column(db.Integer, db.ForeignKey('question.id'))

    choice = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)

    def __repr__(self):
        return '<Answer {}>'.format(self.choice)


class User(UserMixin,db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    answers = db.relationship('Answer',
                            backref='user',
                            lazy='dynamic')

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def number_answers(self):
        return self.answers.count()

    def answered_questions(self):
        return Question.query.join(Answer, Answer.question_id==Question.id).filter(Answer.user_id==self.id).all()

    def has_answered(self,question):
        return self.answers.filter(Answer.question_id == question.id).count()>0

    def next_question(self):
        n_questions = Question.query.count()
        return random.randint(1,n_questions)



class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    example = db.Column(db.String(140))
    system1 = db.Column(db.String(140))
    system2 = db.Column(db.String(140))

    answers = db.relationship('Answer',
                            backref='question',
                            lazy='dynamic')

    def answer(self,choice,user):
        answer = Answer(choice=choice,user_id=user.id,question_id=self.id)
        return answer

    def number_answers(self):
        return self.answers.count()

    def get_filepaths(self):
        target = os.path.join(DATA_PATH,self.example,'target.mp3')
        system1 = os.path.join(DATA_PATH,self.example,self.system1+'.mp3')
        system2 = os.path.join(DATA_PATH,self.example,self.system2+'.mp3')
        return target, system1, system2

    def __repr__(self):
        return '<Question {},{},{}>'.format(self.example,self.system1,self.system2)





# def choose_question_for(user):
#     candidates = Question.query.filter(0<Question.number_answers()<3).all()
#     if candidates == []:
#         question = Question.order_by(func.rand()).limit(1).one()



# class GoldMSIAnswer(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     answer1 = db.Column(db.Integer)
#     timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
#

@login.user_loader
def load_user(id):
    return User.query.get(int(id))
