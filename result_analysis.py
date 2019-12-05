from app import db
from config import MAX_ANSWERS
from app.models import Question, User, Answer

def compute_statistics():
    n_all = Question.query.count()

    n_partial_query = Question.query.filter(db.and_(Question.n_answers>0,Question.n_answers<MAX_ANSWERS))
    n_partial = n_partial_query.count()
    n_answers_array = [0, 0, 0]
    all_partial = n_partial_query.all()
    for question in all_partial:
        n_answers_array[question.n_answers-1] += 1

    n_full_questions =  Question.query.filter(Question.n_answers == MAX_ANSWERS).count()

    n_full_examples = count_full_examples()

    print "Total", n_all
    print "Full examples:", n_full_examples
    print "Full questions: ", n_full_questions
    print "Partial questions:", n_partial, ', distribution:', zip(['1 answer:','2 answers:','3 answers:'],n_answers_array)


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
    for example in partial_examples:

        n = Question.query.filter(Question.example == example).filter(Question.n_answers != MAX_ANSWERS).count()
        if n == 0:
            complete_examples += [example]
        else:
            print example, n

    # print "Complete examples: ", len(complete_examples)
    return len(complete_examples)



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
