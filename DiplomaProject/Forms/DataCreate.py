from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FloatField
from wtforms import validators


class DataCreateForm(FlaskForm):
    country_name = StringField("Name of country: ", [
        validators.DataRequired("Please enter country name."),
        validators.Length(3, 20, "Name should be from 3 to 20 symbols")
    ])
    index_value = FloatField("Index: ", [
                              validators.DataRequired("Please enter float number.")])

    usage_value = FloatField("Usage: ", [
                             validators.DataRequired("Please enter float number")])

    continent_name = StringField("Continent: ", [
        validators.DataRequired("Please enter name of continent."),
        validators.Length(3, 20, "Name should be from 3 to 20 symbols")]
                               )

    submit = SubmitField("Save")

    def check_index(self):
        return bool(self.index_value.data > 0)

    def check_usage(self):
        return bool(self.usage_value.data > 0)
