from flask_wtf import FlaskForm
from wtforms import FormField, FieldList, StringField, TextAreaField, PasswordField, BooleanField, SubmitField, RadioField
from wtforms.validators import ValidationError, DataRequired
from app.models import User



class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Username already in use! Please choose another name')


class EditProfileForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    submit = SubmitField('Submit')

    def __init__(self, original_username, *args, **kwargs):
        super(EditProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username

    def validate_username(self, username):
        if username.data != self.original_username:
            user = User.query.filter_by(username=self.username.data).first()
            if user is not None:
                raise ValidationError('Username already in use! Please choose another name')


class AnswerForm(FlaskForm):
    choice = RadioField('Select your answer:',choices=[(0,'A'),(1,'B')],coerce=int)
    submit = SubmitField('Submit')

    def validate_choice(self,choice):
        if choice.data is None:
            raise ValidationError('Please select an answer!')

class GoldMSIAnswerForm(FlaskForm):
    choice = RadioField(choices=[(1,'A'),(2,'B'),(3,'C'),(4,'D'),(5,'E'),(6,'F'),(7,'G')],coerce=int)
    def validate_choice(self,choice):
        if choice.data is None:
            raise ValidationError('Please select an answer!')

class GoldMSIForm(FlaskForm):
    all_choices = FieldList(FormField(GoldMSIAnswerForm))
    submit = SubmitField('Submit and start experiment')

    def validate_all_choices(self,all_choices):
        for i,entry in enumerate(all_choices.entries):
            print(i, entry.data)
            if entry.data is None:
                raise ValidationError('Please select an answer for question '+str(i)+'!')
