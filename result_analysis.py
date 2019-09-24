from app import db
from config import MAX_ANSWERS
from app.models import Question, User, Answer



def count_answered_questions():
    n_full = Question.query.filter(Question.n_answers == MAX_ANSWERS).count()
    n_all = Question.query.count()

    print "Full answers: ", n_full, " (total:",n_all,')'



def get_complete_questions():
    # q_alias_1 = db.aliased(Question)
    # q_alias_2 = db.aliased(Question)
    # query_full_only = Question.query.join(q_alias,db.and_(q_alias.system1 == Question.system2,q_alias.system2 == Question.system1 ))
    query_ordered = Question.query.filter(Question.system1 < Question.system2).filter(Question.n_answers==MAX_ANSWERS)
    query_reverse = Question.query.filter(Question.system1 > Question.system2).filter(Question.n_answers==MAX_ANSWERS)
    # s = query_reverse.subquery()
    query_both = query_ordered.union(query_reverse)
    q_alias_1 = db.aliased(query_both)
    print query_both.join()
    # all_complete = query_ordered.join(s, q_alias_1.example==s.columns.example).all()

    # query_full_only = Question.query.filter(Question.n_answers == MAX_ANSWERS)
    # questions_full = query_full_only.limit(100).all()
    return query_both.all()

def gather_examples(question_list):
    examples = set()
    for q in question_list:
        examples.add(q.example)
    return list(examples)

def count_full_examples():
    questions_full = get_complete_questions()
    partial_examples=gather_examples(questions_full)

    complete_examples = []
    for example in partial_examples:

        n = Question.query.filter(Question.example == example).filter(Question.n_answers != MAX_ANSWERS).count()
        if n == 0:
            complete_examples += [example]
        else:
            print example, n

    print "Complete examples: ", len(complete_examples)


def gather_ratings(questions,engine,connection):
    # We adopt the convention: systems are always ordered in alphabetical order

    def make_key(question):
        systems_sorted = sorted([question.system1,question.system2])
        return '_'.join([question.example]+systems_sorted)

    def get_rating(question):
        reverse= question.system2 < question.system1
        total = 0
        for answer in question.answers:
            if reverse:
                total += 1-answer.choice
            else:
                total += answer.choice
        return total


    query_full_only = Question.query.filter(questions.columns.n_answers == MAX_ANSWERS)
    questions_full = query_full_only.all()

    ratings = {}
    for question in questions_full:
        rating = get_rating(question)
        try:
            ratings[example] += rating
        except KeyError:
            ratings[example] = rating




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




# count_answered_questions()
# count_full_examples()
print(get_complete_questions())
