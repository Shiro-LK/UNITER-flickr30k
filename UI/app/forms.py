#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:56:53 2021

@author: shiro
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class PredictionForm(FlaskForm):
    text = StringField('Text', validators=[DataRequired()])