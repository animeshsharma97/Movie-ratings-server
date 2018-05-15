
from flask_wtf import FlaskForm
from wtforms import TextAreaField , SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    review = TextAreaField('Your review goes here : ', validators=[DataRequired()])
    submit = SubmitField('Predict rating')
