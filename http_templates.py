#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2019/3/7 19:54
Describe：
    
    
"""

from flask import Flask,render_template

app = Flask(__name__)

@app.route('/temp/<string:name>')
def index(name):
    class Person(object):
        name = u'里斯'
        gender = u'女'
        age = 20
    p = Person()
    context={
        'gender':u'男',
        'age':18,
        'person':p,
        'websites':{
            'baidu':'www.baidu.com',
            'google':'www.google.com',
            'sina':'www.sina.com'
        }
    }
    #表格
    books=[
        {
            'name':u'红楼梦',
            'autor':u'曹雪芹',
            'price':33
        },
        {
            'name': u'水浒传',
            'autor': u'施耐庵',
            'price': 50
        },
        {
            'name': u'西游记',
            'autor': u'吴承恩',
            'price': 43
        },
        {
            'name': u'三国演义',
            'autor': u'罗贯中',
            'price': 60
        }
    ]
    return render_template('index.html', username=name, books=books, **context)


@app.route('/temp/happy')
def happy():
    return render_template('birthday.html')

@app.route('/temp/filter')
def filter():
    books = [
        {
            'name': u'红楼梦',
            'autor': u'曹雪芹',
            'price': 33
        },
        {
            'name': u'水浒传',
            'autor': u'施耐庵',
            'price': 50
        },
        {
            'name': u'西游记',
            'autor': u'吴承恩',
            'price': 43
        },
        {
            'name': u'三国演义',
            'autor': u'罗贯中',
            'price': 60
        }
    ]
    return render_template('filter_.html', books=books)


'''
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
'''