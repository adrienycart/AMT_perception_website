from app import db
from config import MAX_ANSWERS, N_MODELS
from app.models import Question, User, Answer
from  sqlalchemy.sql.expression import func
import numpy as np


def get_avg_goldMSI(answers):
    positive = np.array([1,1,0,1,0,1,1,0,1,0,0,1,1,1,1,1,1,1])
    avg = np.mean(answers*positive + (7-answers)*(1-positive))
    return avg


def parse_answers(user):
    answers = user.gold_msi_answers
    ans_list = answers.split(";")

    gender = str(ans_list[0])
    age = int(ans_list[1])
    disability = bool(ans_list[2])

    gold_msi_answers = np.array(ans_list[3:],dtype=int)
    gold_msi_avg = get_avg_goldMSI(gold_msi_answers)

    return user.id, gender, age, disability,gold_msi_avg, gold_msi_answers


for user in User.query.all():
    print parse_answers(user)
