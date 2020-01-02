#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2018/11/19 16:17
Describe：
    
    
"""
import click
from flask import Flask

app = Flask(__name__)

# the minimal Flask application
@app.route('/')
def index():
    return '<h1>Hello, World!</h1>'