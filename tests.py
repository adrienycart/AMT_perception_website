from datetime import datetime, timedelta
import unittest
from app import app, db
from app.models import User, Question, Answer

class UserModelCase(unittest.TestCase):
    def setUp(self):
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_answer(self):
        u1 = User(username='john')
        u2 = User(username='mary')
        q1 =Question(example="example1",system1="system1",system2="system2")
        q2 =Question(example="example1",system1="system3",system2="system4")
        q3 =Question(example="example2",system1="system1",system2="system2")

        db.session.add(u1)
        db.session.add(u2)
        db.session.add(q1)
        db.session.add(q2)
        db.session.add(q3)
        db.session.commit()

        self.assertEqual(u1.answers.all(), [])
        self.assertEqual(u2.answers.all(), [])

        a1 = q1.answer(0,u1)
        a2 = q2.answer(1,u1)

        a3 = q1.answer(1,u2)
        a4 = q3.answer(0,u2)



        db.session.add(a1)
        db.session.add(a2)
        db.session.add(a3)
        db.session.add(a4)
        db.session.commit()

        print(u1.answered_questions().all())
        self.assertEqual(u1.number_answers(),2)
        self.assertEqual(u2.answered_questions().all(),[q1,q3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
