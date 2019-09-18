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
    recognised = db.Column(db.Boolean)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)

    def __repr__(self):
        return '<Answer {}>'.format(self.choice)


class User(UserMixin,db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    gold_msi_completed = db.Column(db.Boolean, default=False)
    answers = db.relationship('Answer',
                            backref='user',
                            lazy='dynamic')
    # gold_msi_answers = db.relationship('GoldMSIAnswer',backref='user',lazy='dynamic')
    gold_msi_answers = db.Column(db.String(40))

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def number_answers(self):
        return self.answers.count()

    def answered_questions(self):
        return Question.query.join(Answer, Answer.question_id==Question.id).filter(Answer.user_id==self.id).all()

    def has_answered(self,question):
        return self.answers.filter(Answer.question_id == question.id).count()>0

    # def has_answered_gold_msi(self,question):
    #     return self.gold_msi_answers.filter(GoldMSIAnswer.question_id == question.id).count()>0


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


# class GoldMSIAnswer(db.Model):
#     __tablename__ = 'gold_msi_answer'
#
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
#     question_id = db.Column(db.Integer, db.ForeignKey('gold_msi_question.id'))
#
#     choice = db.Column(db.Integer)
#     timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
#
#     def __repr__(self):
#         return '<GoldMSIAnswer {}>'.format(self.choice)
#
#
# class GoldMSIQuestion(db.Model):
#     __tablename__ = 'gold_msi_question'
#
#     id = db.Column(db.Integer, primary_key=True)
#     question = db.Column(db.String(140))
#     choices = db.Column(db.String(140))
#
#     answers = db.relationship('GoldMSIAnswer',
#                             backref='question',
#                             lazy='dynamic')
#
#     def answer(self,choice,user):
#         answer = GoldMSIAnswer(choice=choice,user_id=user.id,question_id=self.id)
#         return answer
#
#     def __repr__(self):
#         return '<GoldMSIQuestion {},{}>'.format(self.question,self.choices)

@login.user_loader
def load_user(id):
    return User.query.get(int(id))
