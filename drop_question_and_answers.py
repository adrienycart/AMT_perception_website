import shutil
import os


# os.remove('app.db')
# shutil.copy2('app.db_save','app.db')


from app import db
from config import MAX_ANSWERS, DATA_PATH
from app.models import Question, User, Answer

for user in User.query.all():
    print user, user.id
    for quest in user.answered_questions():
        print quest.example, quest.system1, quest.system2, quest.n_answers,quest.reverse

print "-----------"
for question in Question.query.filter(Question.n_answers>0):
    print question
    for ans in question.answers.all():
        print ans.user_id, ans.choice, ans.recognised

model_to_drop = 'kelz'


# Save answers

questions_with_answers = Question.query.filter(Question.n_answers>0)

dict_answers = {}

for question in questions_with_answers:
    # print '------'
    # print question
    # print question.answers.all()
    syst_list = [question.system1,question.system2]
    dict_answers[';'.join([question.example,syst_list[0],syst_list[1]])] = [(ans.user_id,syst_list[ans.choice],ans.recognised,ans.timestamp) for ans in question.answers.all()]

# print dict_answers

# Drop answers to drop
for key in dict_answers.keys():
    if model_to_drop in key:
        dict_answers.pop(key)

# print dict_answers

# Erase the whole database (questions and answers only)

Question.query.delete()
Answer.query.delete()

# Re-build the database:
# print('Populate database...')

data_path = os.path.join('app/static',DATA_PATH)
all_folders = [path for path in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,path))]
for folder in all_folders:
    example = folder
    folder_path = os.path.join(data_path,folder)
    files = [elt for elt in os.listdir(folder_path) if elt.endswith('.mp3') and not elt.startswith('.') and not 'target' in elt]
    n_files = len(files)
    for i in range(n_files):
        for j in range(i+1,n_files):
            system1,system2 = sorted([os.path.splitext(files[i])[0],os.path.splitext(files[j])[0]])
            question = Question(example=example,system1=os.path.splitext(files[i])[0],system2=os.path.splitext(files[j])[0])
            db.session.add(question)

db.session.commit()

assert Question.query.count() == 9306


# Add back all answers:

for key, val in dict_answers.items():
    split = key.split(';')

    example,system1,system2 = split
    # print example,system1,system2
    system1,system2 = sorted([system1,system2])
    question = Question.query.filter(db.and_(db.and_(Question.example== example, Question.system1==system1),Question.system2==system2)).all()
    assert len(question)==1
    question = question[0]
    for ans in val:
        user_id,choice,recognised,timestamp = ans
        # print user_id,choice,recognised,timestamp
        # print choice, system1, choice==system1
        choice = int(choice==system2)

        answer = Answer(choice=choice,user_id=user_id,question_id=question.id,recognised=recognised)
        db.session.add(answer)

        db.session.commit()

        # print user_id,choice,recognised,timestamp

    reverse = len(val)%2 == 1
    n_answers = len(val)
    question.n_answers = n_answers
    question.reverse = reverse

db.session.commit()





#
# for question in Question.query.all():
#     assert question.system1 < question.system2
#

print '#####################################'

for user in User.query.all():
    print user
    for quest in user.answered_questions():
        print quest.example, quest.system1, quest.system2, quest.n_answers,quest.reverse

print "-----------"
for question in Question.query.filter(Question.n_answers>0):
    print question
    for ans in question.answers.all():
        print ans.user_id, ans.choice, ans.recognised
