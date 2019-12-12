from app import db
from app import login
import datetime
from flask_login import UserMixin, current_user
import random
import os
from config import DATA_PATH, MAX_ANSWERS, N_MODELS, MIN_DATE, LOCK_TIME,BIAS_MOST_ANSWERS
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

        # Choose in priority questions that might have been left unanswered by current_user
        now = datetime.datetime.utcnow()
        candidate = Question.query.filter(db.and_(Question.ongoing_since>now-LOCK_TIME,Question.ongoing_user==current_user.id)).first()
        # candidate = None
        if candidate is None:

            # Only consider questions that are not being answered right now
            now = datetime.datetime.utcnow()
            candidates = Question.query.filter(db.or_(Question.ongoing_since<now-LOCK_TIME,Question.ongoing_user==current_user.id))

            # Choose a question whose example was never seen by the user
            candidates = candidates.filter(db.not_(Question.example.in_(previously_seen_examples))).order_by(func.random())

            # Among these, rank the examples by number of answers (there is probably a better way to do that in SQL...)
            questions_not_empty = candidates.filter(Question.n_answers>0).all()
            examples_not_empty = set()
            for q in questions_not_empty:
                examples_not_empty.add(q.example)
            examples_not_empty = list(examples_not_empty)
            examples_n_answers = []
            for example in examples_not_empty:
                sum_answers = Question.query.filter(Question.example==example).with_entities(func.sum(Question.n_answers)).scalar()
                # Do not add examples that have been fully answered
                if not sum_answers == MAX_ANSWERS*(N_MODELS*(N_MODELS-1)/2):
                    examples_n_answers += [[example,sum_answers]]
            examples_n_answers = sorted(examples_n_answers,key=lambda x:x[1])
            # print examples_n_answers
            if len(examples_n_answers) == 0:
                # If no answers at all, or if all examples have been fully rated, choose a new, random one
                candidate = None
            else:
                max_tries = 3
                tries = 0
                # Try several times, in case the random example chosen is being answered by someone else
                while candidate is None and tries < max_tries:
                    # print 'TRIES', tries


                    # Choose one of the examples that has most answers (random, but biased towards more answers)
                    example_idx = int(len(examples_n_answers) * pow(random.random(),BIAS_MOST_ANSWERS))
                    # In case BIAS_MOST_ANSWERS==0:
                    if example_idx == len(examples_n_answers):
                        example_idx = len(examples_n_answers)-1
                    chosen_example = examples_n_answers[example_idx][0]
                    # Choose a question for that example
                    candidates_example=Question.query.filter(Question.example==chosen_example)
                    # print candidates_example.all()
                    # Among these, only consider questions that are not being answered
                    candidates_example = candidates_example.filter(db.or_(Question.ongoing_since<now-LOCK_TIME,Question.ongoing_user==current_user.id))
                    # Among these, choose a question that was already answers, but still lacks some:
                    candidate = candidates_example.filter(db.and_(Question.n_answers>0,Question.n_answers<MAX_ANSWERS)).order_by(func.random()).first()
                    # print candidate
                    # If all questions are fully answered, choose a new question for this example
                    if candidate is None:
                        # print "New question for same example"
                        candidate = candidates_example.filter(Question.n_answers<MAX_ANSWERS).order_by(func.random()).first()
                    tries += 1

            # If there was no suitable candidate choose a random one.
            # (e.g. no answers yet, or all questions have been fully answered,
            # or the only questions left have all been seen by current_user,
            # or the only candidates are all being answered by other people...)
            if candidate is None:
                # print("Picking new example")
                candidate = candidates.filter(Question.n_answers<MAX_ANSWERS).order_by(func.random()).first()
        # print candidate
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
