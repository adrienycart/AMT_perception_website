import sqlalchemy as db
from config import MAX_ANSWERS


def fetch_all(query, engine,connection):
    ResultProxy = connection.execute(query)
    ResultSet = ResultProxy.fetchall()
    return ResultSet


def count_answered_questions(questions,engine,connection):
    query_full_only = db.select([questions]).where(questions.columns.n_answers == MAX_ANSWERS).count()
    n_full = fetch_all(query_full_only,engine,connection)[0][0]

    query_all = db.select([questions]).count()
    n_all = fetch_all(query_all,engine,connection)[0][0]

    print "Full answers: ", n_full, " (total:",n_all,')'


def count_full_examples(questions,engine,connection):
    query_full_only = db.select([questions]).where(questions.columns.n_answers == MAX_ANSWERS)
    questions_full = fetch_all(query_full_only,engine,connection)
    partial_examples = set()
    for q in questions_full:
        partial_examples.add(q.example)
    partial_examples=list(partial_examples)

    complete_examples = []
    for example in partial_examples:
        query = db.select([questions]).where(questions.columns.example == example).where(questions.columns.n_answers != MAX_ANSWERS).count()
        n = fetch_all(query,engine,connection)[0][0]
        if n == 0:
            complete_examples += [example]
        else:
            print example, n

    print "Complete examples: ", len(complete_examples)






engine = db.create_engine('sqlite:///app.db')
connection = engine.connect()

metadata = db.MetaData()
answers = db.Table('Answer', metadata, autoload=True, autoload_with=engine)
users = db.Table('User', metadata, autoload=True, autoload_with=engine)
questions = db.Table('Question', metadata, autoload=True, autoload_with=engine)
# print(answers.columns.keys())
# query = db.select([answers])
# ResultProxy = connection.execute(query)
# ResultSet = ResultProxy.fetchall()
# print(ResultSet)
#
# print(users.columns.keys())
# query = db.select([users])
# ResultProxy = connection.execute(query)
# ResultSet = ResultProxy.fetchall()
# print(ResultSet)


count_answered_questions(questions,engine,connection)
count_full_examples(questions,engine,connection)
