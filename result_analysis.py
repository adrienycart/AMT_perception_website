from app import db
from config import MAX_ANSWERS, N_MODELS
from app.models import Question, User, Answer
from  sqlalchemy.sql.expression import func

def compute_statistics():
    n_all = Question.query.count()

    # Partial questions
    n_partial_query = Question.query.filter(db.and_(Question.n_answers>0,Question.n_answers<MAX_ANSWERS))
    n_partial = n_partial_query.count()
    partial_questions_array = [0, 0, 0]
    all_partial = n_partial_query.all()
    for question in all_partial:
        partial_questions_array[question.n_answers-1] += 1

    n_full_questions =  Question.query.filter(Question.n_answers == MAX_ANSWERS).count()

    # Partial examples
    n_full_examples, partial_examples = count_full_examples()
    partial_examples_array = [0, 0, 0, 0, 0]
    for example in partial_examples:
        partial_examples_array[5-example[1]] += 1

    # Non-empty and non-full examples:
    questions_not_empty = Question.query.filter(Question.n_answers>0).all()
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

    n_answers = Answer.query.count()
    n_participants = User.query.count()

    print "Total", n_all
    print "Full examples:", n_full_examples
    print "Partial examples:", len(partial_examples), ', distribution:', zip(['1 question:','2 questions:','3 questions:','4 questions:','5 questions:'],partial_examples_array)
    print "Non-empty examples:", len(examples_n_answers)
    print "Full questions: ", n_full_questions
    print "Partial questions:", n_partial, ', distribution:', zip(['1 answer:','2 answers:','3 answers:'],partial_questions_array)
    print 'Total answers:', n_answers
    print 'Total participants:', n_participants

def count_answered_questions():
    n_full = Question.query.filter(Question.n_answers == MAX_ANSWERS).count()
    n_all = Question.query.count()

    print "Full answers: ", n_full, " (total:",n_all,')'

def get_users(with_answers=False):
    all_users = User.query.all()
    user_answers = []
    for user in all_users:
        username = user.username
        n_answers = user.number_answers()
        comment = user.comments
        print 'User:', username, " n. answers:", n_answers
        # print user.gold_msi_answers
        if comment is not None:
            print '    ', comment
        if with_answers:
            for ans in user.answers:
                print '    ', Question.query.get(ans.question_id), ans

        user_answers += [[username,n_answers]]
    return user_answers

def get_complete_questions():
    query = Question.query.filter(Question.n_answers==MAX_ANSWERS)
    return query.all()

def gather_examples(question_list):
    examples = set()
    for q in question_list:
        examples.add(q.example)
    return list(examples)

def count_full_examples():
    questions_full = get_complete_questions()
    partial_examples=gather_examples(questions_full)

    complete_examples = []
    partial_examples_number = []
    for example in partial_examples:

        n = Question.query.filter(Question.example == example).filter(Question.n_answers != MAX_ANSWERS).count()
        if n == 0:
            complete_examples += [example]
        else:
            partial_examples_number += [[example, n]]

    # print "Complete examples: ", len(complete_examples)
    return len(complete_examples), partial_examples_number



def gather_ratings():
    # We adopt the convention: systems are always ordered in alphabetical order

    def make_key(question):
        systems_sorted = sorted([question.system1,question.system2])
        return '_'.join([question.example]+systems_sorted)

    def get_rating(question):
        total = 0
        for answer in question.answers:
            print answer
            total += answer.choice
        return total



    questions_full = get_complete_questions()

    ratings = {}
    for question in questions_full:
        rating = get_rating(question)
        key = make_key(question)
        ratings[key] = rating
    return ratings





# engine = db.create_engine('sqlite:///app.db')
# connection = engine.connect()
#
# metadata = db.MetaData()
# answers = db.Table('Answer', metadata, autoload=True, autoload_with=engine)
# users = db.Table('User', metadata, autoload=True, autoload_with=engine)
# questions = db.Table('Question', metadata, autoload=True, autoload_with=engine)
# # print(answers.columns.keys())
# # query = db.select([questions])
# # ResultProxy = connection.execute(query)
# # ResultSet = ResultProxy.fetchall()
#
#
# question = get_complete_questions(questions, engine, connection)[0]
# print question.answers

# print(ResultSet)
#
# print(users.columns.keys())
# query = db.select([users])
# ResultProxy = connection.execute(query)
# ResultSet = ResultProxy.fetchall()
# print(ResultSet)



# print(Answer.query.all())
# print(count_answered_questions())
# # count_full_examples()
# print(get_complete_questions())
# print(count_full_examples())
# print(gather_ratings())
compute_statistics()
get_users()
