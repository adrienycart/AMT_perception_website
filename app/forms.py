from flask_wtf import FlaskForm
from wtforms import IntegerField, FormField, FieldList, StringField, TextAreaField, PasswordField, BooleanField, SubmitField, RadioField
from wtforms.validators import ValidationError, DataRequired, Length
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
    comments = TextAreaField('Tell us what you think!', validators=[Length(min=0, max=1000)])
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
    known = BooleanField("I know this piece")
    difficulty = RadioField('How easy was it?',choices=[(1,'A'),(2,'B'),(3,'C'),(4,'D'),(5,'E')],coerce=int)
    submit = SubmitField('Next question')

    # def validate_choice(self,choice):
    #     if choice.data is None:
    #         raise ValidationError('Please select an answer!')


class GoldMSIAnswerForm(FlaskForm):
    choice = RadioField(choices=[(1,'A'),(2,'B'),(3,'C'),(4,'D'),(5,'E'),(6,'F'),(7,'G')],coerce=int)
    # def validate_choice(self,choice):
    #     if choice.data is None:
    #         raise ValidationError('Please select an answer!')

class GoldMSIForm(FlaskForm):

    gender = RadioField(choices=[('male','Male'),('female','Female'),('other','Non-binary')],coerce=str)
    age = IntegerField(label='Age')

    disability = BooleanField('I have a hearing disability')

    all_choices = FieldList(FormField(GoldMSIAnswerForm))
    submit = SubmitField('Submit and start experiment')

    def validate_all_choices(self,all_choices):
        for i,entry in enumerate(all_choices.entries):
            if entry.data is None:
                raise ValidationError('Please select an answer for question '+str(i)+'!')


class ConsentForm(FlaskForm):
    consent = FieldList(BooleanField())
    submit = SubmitField('Submit')


    def validate_consent(self,consent):
        if len(consent.entries)<4 or not all([entry.data for entry in consent.entries]):
            raise ValidationError('Please accept all!')
