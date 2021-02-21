#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:55:26 2021

@author: shiro
"""

from flask import Flask



app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "/static/images/"
app.config['SECRET_KEY'] = "123"
from app import routes