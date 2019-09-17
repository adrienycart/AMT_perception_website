from app import db
from app import login
from datetime import datetime
from flask_login import UserMixin



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
        answers = self.answers.all()
        questions = []
        for answer in answers:
            questions += [answer.question]
        return questions
        # return self.answers.question

    def has_answered(self,question):
        return self.answers.filter(Answer.question_id == question.id).count()>0




class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    example = db.Column(db.String(140))
    system1 = db.Column(db.String(140))
    system2 = db.Column(db.String(140))
    example_id = db.Column(db.Integer)
    system1_id = db.Column(db.Integer)
    system2_id = db.Column(db.Integer)
    answers = db.relationship('Answer',
                            backref='question',
                            lazy='dynamic')

    def answer(self,choice,user):
        answer = Answer(choice=choice,user_id=user.id,question_id=self.id)
        return answer

    def __repr__(self):
        return '<Question {},{},{}>'.format(self.example,self.system1,self.system2)




# class GoldMSIAnswer(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     answer1 = db.Column(db.Integer)
#     timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
#
# @login.user_loader
# def load_user(id):
#     return User.query.get(int(id))
