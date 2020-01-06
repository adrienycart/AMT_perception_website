from app import db
from config import MAX_ANSWERS, N_MODELS
from app.models import Question, User, Answer
from  sqlalchemy.sql.expression import func
import numpy as np


def get_avg_goldMSI(answers):
    positive = np.array([1,1,0,1,0,1,1,0,1,0,0,1,1,1,1,1,1])
    avg = np.mean(answers*positive + (7-answers)*(1-positive))
    return avg


def parse_answers(user):
    answers = user.gold_msi_answers
    ans_list = answers.split(";")

    gender = str(ans_list[0])
    age = int(ans_list[1])
    disability = ans_list[2]=='True'

    # DROP LAST ANSWER, IT WAS NOT RECORDED FOR FEMALE PARTICIPANTS
    gold_msi_answers = np.array(ans_list[3:-1],dtype=int)
    gold_msi_avg = get_avg_goldMSI(gold_msi_answers)

    return [user.username, len(user.answers.all()), gender, age, disability,gold_msi_avg]+ list(gold_msi_answers)



# USER DATA
user_data = []
user_data += [["user","n_answers","gender","age","disability","gold_msi_avg"]+["gold_msi_"+str(i) for i in range(17)]]
for user in User.query.all():
    if user.number_answers() > 0:
        user_data += [parse_answers(user)]

user_data = np.array(user_data,dtype=object)
np.savetxt("user_data.csv",user_data, fmt="%s")


# ANSWERS DATA

answers_data=[]
answers_data+=[["question_id","example",'system1','system2','user_id',"answer",'recognised','difficulty','time']]

for question in Question.query.filter(Question.n_answers==MAX_ANSWERS).all():
    example = question.example
    system1 = question.system1
    system2 = question.system2

    for answer in question.answers.all():
        answers_data += [[question.id,example,system1,system2,answer.user_id,answer.choice,answer.recognised,answer.difficulty, answer.time_taken]]

answers_data = np.array(answers_data,dtype=object)
# print answers_data
np.savetxt("answers_data.csv",answers_data, fmt="%s")

# COMMENTS
comments = ""
for user in User.query.all():
    if user.comments is not None and len(user.comments) > 0:
        comments += '-----------------------\n'
        comments += str(user.id)
        comments += "\n"
        comments += user.comments
        comments += "\n"

text_file = open("comments.txt", "w")
n = text_file.write(comments)
text_file.close()
