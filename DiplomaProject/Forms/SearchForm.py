from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, SelectField
from wtforms import validators


class SearchForm(FlaskForm):
    type_field = SelectField('Choose criterion of search:', choices=[
        ('country_name', 'Name of country'),
        ('continent_name', 'Name of continent'),
    ])
    search_value = StringField("Value: ", [validators.DataRequired('shouldnt be empty value')])

    submit = SubmitField("Search")