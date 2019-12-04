from app import db
from app import login
import datetime
from flask_login import UserMixin, current_user
import random
import os
from config import DATA_PATH, MAX_ANSWERS, MIN_DATE, LOCK_TIME
from  sqlalchemy.sql.expression import func




class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    question_id = db.Column(db.Integer, db.ForeignKey('question.id'))

    choice = db.Column(db.Integer)
    recognised = db.Column(db.Boolean)
    difficulty = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.datetime.utcnow)
    time_taken =  db.Column(db.Float)

    def __repr__(self):
        return '<Answer {},known {},difficulty {},time {}>'.format(self.choice,self.recognised, self.difficulty, self.time_taken)


class User(UserMixin,db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    last_seen = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    ethics_approved = db.Column(db.Boolean, default=False)
    gold_msi_completed = db.Column(db.Boolean, default=False)
    answers = db.relationship('Answer',
                            backref='user',
                            lazy='dynamic')
    # gold_msi_answers = db.relationship('GoldMSIAnswer',backref='user',lazy='dynamic')
    gold_msi_answers = db.Column(db.String(50))
    comments = db.Column(db.String(1000))

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def number_answers(self):
        return self.answers.count()

    def answered_questions(self):
        return Question.query.join(Answer, Answer.question_id==Question.id).filter(Answer.user_id==self.id).all()


    def answered_questions_with_answers(self):
        answers = self.answers
        questions = []
        for answer in answers:
            questions += [answer.question]
        return zip(questions, answers)


    def has_answered(self,question):
        return self.answers.filter(Answer.question_id == question.id).count()>0

    def next_question(self):

        previously_seen_examples = set()
        for q in self.answered_questions():
            previously_seen_examples.add(q.example)
        previously_seen_examples = list(previously_seen_examples)

        # Only consider questions that are not being answered right now
        now = datetime.datetime.utcnow()
        candidates = Question.query.filter(db.or_(Question.ongoing_since<now-LOCK_TIME,Question.ongoing_user==current_user.id))
        # print(candidates.all())

        # print Question.query.filter(Question.ongoing_since>now-LOCK_TIME).all()
        # print Question.query.filter(Question.ongoing_user==current_user.id).all()

        # Choose a question whose example was never seen by the user and that is not fully answered
        candidates = candidates.filter(db.not_(Question.example.in_(previously_seen_examples))).order_by(func.random())
        # print(candidates.all())
        # print(Question.query.first().n_answers)

        # Among these, choose a question that was already answers, but still lacks some:
        candidate = candidates.filter(db.and_(Question.n_answers>0,Question.n_answers<MAX_ANSWERS)).order_by(func.random()).first()


        # If no question fullfills that criterion, choose a question such that
        # its example was already evaluated for some other systems (still not previously seen)
        if candidate is None:
            # print("Trying to find a partially-filled example")
            partial_examples = set()
            for q in candidates.filter(Question.n_answers==MAX_ANSWERS):
                partial_examples.add(q.example)
            partial_examples=list(partial_examples)
            candidate = candidates.filter(Question.example.in_(partial_examples)).filter(Question.n_answers<MAX_ANSWERS).order_by(func.random()).first()
            # If no question fullfills that criterion, choose any question with
            # unseen example, and lacking answers (it should be an example seen by no-one yet)
            if candidate is None:
                # print("Picking new example")
                candidate = candidates.filter(Question.n_answers<MAX_ANSWERS).order_by(func.random()).first()

        return candidate.id



class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    example = db.Column(db.String(140))
    system1 = db.Column(db.String(140))
    system2 = db.Column(db.String(140))

    n_answers = db.Column(db.Integer,default=0)
    reverse = db.Column(db.Boolean, default=False)
    ongoing_user = db.Column(db.Integer,default=-1)
    ongoing_since = db.Column(db.DateTime,default=MIN_DATE)

    answers = db.relationship('Answer',
                            backref='question',
                            lazy='dynamic')

    def answer(self,choice,user,recognised,difficulty):
        time_taken = datetime.datetime.utcnow()-self.ongoing_since
        time_taken_s = time_taken.total_seconds()
        choice = 1-choice if self.reverse else choice
        answer = Answer(choice=choice,user_id=user.id,question_id=self.id,recognised=recognised,difficulty=difficulty,time_taken=time_taken_s)
        db.session.add(answer)
        self.n_answers += 1
        self.reverse = not self.reverse
        self.ongoing_since = MIN_DATE
        self.ongoing_user = -1
        db.session.commit()
        return

    def get_filepaths(self):
        target = os.path.join(DATA_PATH,self.example,'target.mp3')
        system1 = os.path.join(DATA_PATH,self.example,self.system1+'.mp3')
        system2 = os.path.join(DATA_PATH,self.example,self.system2+'.mp3')
        if self.reverse:
            return target, system2, system1
        else:
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
#     timestamp = db.Column(db.DateTime, index=True, default=datetime.datetime.utcnow)
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
