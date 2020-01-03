from flask import Flask, request, Response, send_from_directory, jsonify, render_template, redirect, url_for, session, \
    make_response
import logging
import json
import os
from SensorData import SensorData
import imp
from werkzeug.utils import secure_filename
import traceback
import zipfile
import platform
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import os
import shutil
from sklearn import preprocessing, linear_model
import matplotlib as mpl
from matplotlib import pyplot as plt
import joblib
import math
from datetime import datetime
from common import O_COLUMNS_E, O_COLUMNS, O_COLUMNS_ACC, N_COLUMNS, N_COLUMNS_SPE, O_COLUMNS_NINE

import warnings
import test

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

app = Flask(__name__)
app.config.from_object(test)
app.secret_key = 'super secret key'
app.config['SECRET_KEY'] = '123456'
handler = logging.FileHandler('log/flask.log')
app.logger.addHandler(handler)
app.config['JSON_AS_ASCII'] = False

# 定义动作特征降维的维度
ACTION_COMPONENTS = 10
# HMM 预测得分阈值
HMM_SCORE_MAX = -5
# 提交第一层数组长度
FIRST_ACTION_DATA_WINDOW_SIZE = 150
# 定义第一层数组窗口的滑动的距离
MOVIE_SIZE = 20
# 定义第二层窗口启动长度
SECOND_ACTION_DATA_BEGIN_SIZE = 20
# 第三层数组长度（需要识别的动作数组）
ACTION_WINDOW_SIZE = 30
# 定义探测数组的长度（是否开始运动）
ACTION_DETECT_ACTION_BEGIN_SIZE = -3
# 定义探测数组的斜率阈值
ACTION_DETECT_K_START_VALUE = 4
ACTION_DETECT_K_END_VALUE = -2

# 第一层窗口
action_data_window_queue = DataFrame(columns=O_COLUMNS_ACC)
# 第二层窗口
second_action_data_queue = DataFrame(columns=O_COLUMNS_ACC)
isActionDetectStart = False  # 动作是否开始
isActionDetectEnd = False  # 动作是否结束
actionDetectEndCounter = 0  # 动作结束计数器

# 复合数据
complex_data = DataFrame(np.array(pd.read_csv('src/pre/action_feature/all_feature_data_X.csv'))[:, 1:-1],
                         columns=N_COLUMNS)

# 保存实时识别的动作
ACTION_DATA_FILE = 'static/data/actionDataRecognition/0.csv'


@app.route('/windows')
def windows():
    print('下载Windows10')
    file = 'kms.rar'
    if os.path.exists(r'D:\home\temp\cn_windows_10.iso'):
        return send_from_directory(r'D:\home\temp', file)
    else:
        return 'error'


@app.route('/')
def hello_world():
    strLog = str('LOGGING:' + '-- ' + str(
        datetime.now()) + " -- " + str(request.remote_addr) + ' -- ' + str(request.url) + ' -- ' + str(
        request.headers.get("User-Agent")))
    try:
        with open(r'log/flask.log', 'a+') as f:
            f.write(strLog + '\n')
    except Exception as e:
        print(traceback.print_exc())
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def hello_login():
    if request.method == "GET":
        return render_template('login.html')
        # return redirect(url_for('.hello_main'))
    else:
        user = request.form.get('username')
        pwd = request.form.get('password')
        if user == 'swulab' and pwd == 'swulab':
            try:
                session['user_info'] = user
            except Exception as e:
                print(e)
            finally:
                return redirect('/main')
        else:
            return render_template('login.html')


# 定义一个装饰器，用来做登录验证
def wapper(func):
    def inner(*args, **kwargs):
        if not session.get('user_info'):
            return redirect('/login')
        return func(*args, **kwargs)

    return inner


# 视图函数可以返回不同的状态码,400为无效
@app.route('/main')
@wapper
def hello_main():
    class ApiClass(object):
        def __init__(self, uid, disc, api):
            self.uid = uid
            self.disc = disc
            self.api = api

    api1 = ApiClass(1, '采集数据下载', request.url_root + 'action/download/collection')
    api2 = ApiClass(2, '实时识别数据下载', request.url_root + 'action/download/recognition')
    api3 = ApiClass(3, '采集数据上传（页面）', request.url_root + 'action/upload')
    api4 = ApiClass(4, '备份数据', request.url_root + 'action/backupsFile')
    api5 = ApiClass(5, '恢复数据', request.url_root + 'action/recoverBackups')
    api6 = ApiClass(6, '实时识别并保存数据', request.url_root + 'action/greatRecognition')
    api7 = ApiClass(7, '查看数据备份', request.url_root + 'action/download/backups')
    api8 = ApiClass(8, '添加实时识别日期文件', request.url_root + '/action/addfile')
    api9 = ApiClass(9, '下载日志', request.url_root + 'user/download/logs')

    apiList = [api1, api2, api3, api4, api5, api6, api7, api8, api9]
    return render_template('apiPage.html', apiList=apiList)


@app.route('/user/<name>')
def hello_user(name):
    return 'hello,{}'.format(name)


# 下载日志文件
@app.route('/user/download/logs', methods=['GET'])
def downloadLogsWithTemp():
    logPath = []
    for maindir, subdir, file_name_list in os.walk('log'):
        for filename in file_name_list:
            logPath.append(filename)
    return render_template('downLogs.html', logPath=logPath)


# 下载日志文件
@app.route('/user/downloadLogs/<string:fileName>')
def downloadLogs(fileName):
    print(fileName)
    return send_from_directory('log', fileName)


# 删除日志文件
@app.route('/user/deleteLogs/<string:fileName>')
def deleteLogs(fileName):
    filePath = os.path.join('log', fileName)
    if os.path.exists(filePath):
        os.remove(filePath)
        return redirect(url_for('downloadLogsWithTemp'))
    else:
        return 'file not exits'


# 在线查看日志
@app.route('/user/view/log/<string:fileName>')
def viewLogDetail(fileName):
    try:
        resp = make_response(open(os.path.join('log', fileName)).read())
        resp.headers["Content-type"] = "text/plan;charset=UTF-8"
    except:
        resp = 'File format error '
    return resp


# 上传数据（模拟采集数据上传）页面（测试）
@app.route('/action/upload')
def upload():
    msg = request.args.get('message')
    return render_template('upAndDownFile.html', message=msg)


#  处理上传请求
@app.route('/action/uploadData', methods=['GET', 'POST'])
def uploadDataCollection():
    message = 'error'
    basepath = os.path.dirname(__file__)  # 当前文件所在路径

    namePath = request.form.get('namePath')
    if namePath == None: namePath = 'test'

    actionPath = request.form.get('actionPath')
    if actionPath == None: actionPath = 'action1'

    datePath = request.form.get('datePath')
    if datePath == None: datePath = '00001.csv'

    # 组合保存文件的路径
    filePath = namePath + '/' + actionPath + '/' + datePath
    filePath = os.path.join(basepath, 'static/data/actionDataCollection/' + filePath)
    if not os.path.exists(filePath):
        os.makedirs(filePath)

    if request.method == 'POST':
        file_obj = request.files.getlist('file')
        for f in file_obj:
            # secure_filename(f.filename) 获取文件后缀
            upload_path = os.path.join(filePath, f.filename)
            f.save(upload_path)
            message = 'success'

    # return redirect(url_for('.upload', message=message))
    return message


# 下载 实时识别 保存的数据
@app.route('/action/download/recognition', methods=['GET'])
def downloadRecognition():
    file_list = all_path('static/data/actionDataRecognition')
    return render_template('downDataRecognition.html', fileList=file_list)
    '''
    new_fileList = []
    for i in range(10):
        for file in file_list:
             new_fileList.append(file)
     '''


# 下载实时识别文件
@app.route('/action/download/recognition/<string:fileName>')
def downloadRecognitionFiles(fileName):
    outFilePath = r'static/data/actionDataRecognition'
    # print(fileName)
    if os.path.exists(os.path.join(outFilePath, fileName)):
        return send_from_directory(outFilePath, fileName)
    else:
        return json.dumps({'error': '文件路径不存在'}, ensure_ascii=False)


# 下载数据采集的文件
@app.route('/action/download/collection', methods=['GET'])
def downloadCollection():
    path = 'static/data/actionDataCollection'
    # path = 'static'
    if not os.path.exists(path):
        return json.dumps({'error': ' 系统找不到指定的路径'}, ensure_ascii=False)

    parent = os.listdir(path)
    dirList = []
    fileList = []

    for child in parent:
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            dirList.append(child)
        else:
            fileList.append(child)
    treePath = {'dirList': dirList, 'fileList': fileList}
    return render_template('downDataCollection.html', treePath=treePath)
    # pass


# 查询数据采集的文件及路径下的文件
@app.route('/action/download/query')
def queryCollectionPath():
    pathDir = request.args.get("pathDir")
    path = 'static/data/actionDataCollection'
    # path = 'static'
    # http://127.0.0.1:5000/action/download/collection/query?pathDir=actionDataCollection,testUserName,testAction,testDate
    for p in pathDir.split(','):
        path = path + '/' + p
    if not os.path.exists(path):
        return json.dumps({'error': ' 系统找不到指定的路径'}, ensure_ascii=False)

    parent = os.listdir(path)
    dirList = []
    fileList = []

    for child in parent:
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            dirList.append(child)
        else:
            fileList.append(child)

    # 一级目录，根据action排序
    if len(pathDir.split(',')) == 1:
        dirList.sort(key=lambda x: int(x[6:]))
        fileList.sort(key=lambda x: int(x[:-4]))
    # 二级，根据日期查询
    if len(pathDir.split(',')) == 2:
        dirList.sort(key=lambda x: int(x))
        fileList.sort(key=lambda x: int(x[:-4]))
    # 三级
    if len(pathDir.split(',')) == 3:
        dirList.sort(key=lambda x: int(x))
        fileList.sort(key=lambda x: int(x[:-4]))

    treePath = {'dirList': dirList, 'fileList': fileList}
    return json.dumps(treePath, ensure_ascii=False)


# 实时识别 CSV 数据展示
@app.route('/action/chart/recognition/<string:fileName>')
def dataChartShowFromRecognition(fileName):
    filePath = 'static/data/actionDataRecognition/' + fileName
    if os.path.exists(filePath):
        dataMat = pd.read_csv(filePath, names=O_COLUMNS_ACC)
        # print(dataMat)
        dataAcc = list(dataMat['ACC'])
        xAxis = []
        for index in range(len(dataAcc)):
            xAxis.append(index)
        content = {'file': fileName, 'xaxis': xAxis, 'data': dataAcc}

        return render_template('dataChart.html', content=content)
    else:
        pass

# 数据采集 CSV 数据展示
@app.route('/action/chart/collection/<string:pathDir>')
def dataChartShowFromCollection(pathDir):
    filePath = 'static/data/actionDataCollection'
    for p in pathDir.split(','):
        filePath = filePath + '/' + p
    if os.path.exists(filePath):
        dataMat = pd.read_csv(filePath, names=O_COLUMNS_NINE)

        # dataMat['ACC'] = dataMat.apply(lambda row: float(row['AX']) + float(row['AY']) + float(row['AZ']), axis=1)
        dataMat['ACC'] = dataMat.apply(
            lambda row: math.sqrt(float(row['AX']) ** 2 + float(row['AY']) ** 2 + float(row['AZ'] ** 2)), axis=1)

        # 计算  根号下（ax²+ay²+az²）= 重力加速度
        dataMat['ag'] = dataMat.apply(
            lambda row: math.sqrt(float(row['AX']) ** 2 + float(row['AY']) ** 2 + float(row['AZ']) ** 2), axis=1)
        ag = dataMat['ag'].mean()  # 平均加速度
        dataAcc = list(dataMat['ACC'])
        xAxis = []
        for index in range(len(dataAcc)):
            xAxis.append(index)
        content = {'file': pathDir.split(',')[-1], 'xaxis': xAxis, 'data': dataAcc, 'ag': ag}

        return render_template('dataChart.html', content=content)
    else:
        pass


# 删除文件
@app.route('/action/delete/<string:fileName>')
def deleteFileWithName(fileName):
    filePath = 'static/data/actionDataRecognition/' + fileName
    if os.path.exists(filePath):
        os.remove(filePath)
        return redirect(url_for('.downloadRecognition'))
    else:
        return json.dumps({'error': '文件路径不存在'}, ensure_ascii=False)
    # return 文件不存在
    # return send_from_directory(app.config['DOWNLOAD_PATH'], 'static/actionDataRecognition/' + fileName, as_attachment=True)
    # return app.send_static_file('static/imgae/image.jpg')


# 备份服务器数据
@app.route('/action/backupsFile')
def backupsFile():
    try:
        if platform.system() == 'Windows':
            sourceSrcDir = os.path.abspath(r'D:\home\developer\PycharmProject\ActionCloudService\static\data')
            dstSrcDir = os.path.abspath(r'D:\temp\backups\data0')
        else:
            sourceSrcDir = os.path.abspath(r'/home/yan/dev/ActionCloudService/static/data')
            dstSrcDir = os.path.abspath(r'/home/yan/backups/data0')

        dirNum = 0
        if os.path.exists(dstSrcDir):
            rootDir = dstSrcDir[:-5]
            for filename in os.listdir(rootDir):
                if os.path.isdir(rootDir + filename):
                    dirNum += 1
            dstSrcDir = rootDir + 'data' + str(dirNum)

            # dstSrcDir = destStrDirList[:-1]+'\\'+destStrDirList[-1]+''
        shutil.copytree(sourceSrcDir, dstSrcDir)
        success = {'message': '备份成功' + dstSrcDir}
    except Exception as e:
        success = {'message': '备份失败'}
    return json.dumps(success, ensure_ascii=False)


# 恢复备份数据
@app.route('/action/recoverBackups/<string:dirPath>')
def recoverBackups(dirPath):
    try:
        if platform.system() == 'Windows':
            sourceSrcDir = os.path.abspath(r'D:\temp\backups')
            dstSrcDir = os.path.abspath(r'D:\home\developer\PycharmProject\ActionCloudService\static\data')
        else:
            sourceSrcDir = os.path.abspath(r'/home/yan/backups')
            dstSrcDir = os.path.abspath(r'/home/yan/dev/ActionCloudService/static/data')
        '''
        dirNum = 0
        if os.path.exists(sourceSrcDir):
            rootDir = sourceSrcDir[:-5]
            for filename in os.listdir(rootDir):
                if os.path.isdir(rootDir + filename):
                    dirNum += 1
            sourceSrcDir = rootDir + 'data' + str(dirNum - 1)
        '''

        sourceSrcDir = os.path.join(sourceSrcDir, dirPath)
        if os.path.exists(dstSrcDir):
            shutil.rmtree(dstSrcDir)

        shutil.copytree(sourceSrcDir, dstSrcDir)
        message = '恢复成功' + dirPath
    except Exception as e:
        message = '恢复失败' + dirPath
    return redirect(url_for('downloadBackups', msg=message))


# 下载备份数据（页面）
@app.route('/action/download/backups')
def downloadBackups():
    message = ''
    if len(request.url) > 45:
        message = request.url.split('?')[1][4:]
    if platform.system() == 'Windows':
        backupsDir = os.path.abspath(r'D:\temp\backups')
    else:
        backupsDir = os.path.abspath(r'/home/yan/backups')

    if not os.path.exists(backupsDir):
        return json.dumps({'error': ' 系统找不到指定的路径'}, ensure_ascii=False)

    parentPath = os.listdir(backupsDir)  # 子级文件夹
    parentPath.sort(key=lambda x: int(x[4:]))  # 排序
    dirNameList = []
    for path in parentPath:
        if os.path.isdir(os.path.join(backupsDir, path)):
            dirNameList.append(path)
    return render_template('downDataBackups.html', dirNameList=dirNameList, msg=message)


# 压缩备份文件，并下载
@app.route('/action/download/backupswithzip/<string:dirName>')
def downloadBackupsWithZip(dirName):
    if platform.system() == 'Windows':
        outFullName = os.path.abspath(r'D:\temp\backupsZip')
        inputFullName = os.path.abspath(r'D:\temp\backups')
    else:
        outFullName = os.path.abspath(r'/home/yan/backupsZip')
        inputFullName = os.path.abspath(r'/home/yan/backups')

    if not os.path.exists(outFullName):
        os.makedirs(outFullName)

    input_path = os.path.join(inputFullName, dirName)  # 目标（压缩）文件夹路径
    out_path = os.path.join(outFullName, dirName + '.zip')  # 输出压缩文件
    if not os.path.exists(input_path):
        return json.dumps({'error': ' 系统找不到指定的路径'}, ensure_ascii=False)

    # 压缩文件不存在的情况
    if not os.path.exists(out_path):
        try:
            z = zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED)  # 参数一：文件夹名
            for dirpath, dirnames, filenames in os.walk(input_path):
                fpath = dirpath.replace(input_path, '')  # 这一句很重要，不replace的话，就从根目录开始复制
                fpath = fpath and fpath + os.sep or ''  # 这句话理解我也点郁闷，实现当前文件夹以及包含的所有文件的压缩
                for filename in filenames:
                    z.write(os.path.join(dirpath, filename), fpath + filename)
                    # print('压缩成功')
            z.close()
            # 直接下载文件
            return send_from_directory(outFullName, dirName + '.zip')
        except Exception as e:
            return json.dumps({'error': ' 未知错误'}, ensure_ascii=False)
    else:
        # 直接下载文件
        return send_from_directory(outFullName, dirName + '.zip')


@app.route('/json')
def foo():
    return jsonify(name='yan', gender='man'), 200


# 获取指定文件夹下的文件夹及文件
def get_zip_file(input_path, result):
    """
    对目录进行深度优先遍历
    :param input_path:
    :param result:
    :return:
    """
    files = os.listdir(input_path)
    for file in files:
        if os.path.isdir(input_path + '/' + file):
            get_zip_file(input_path + '/' + file, result)
        else:
            result.append(input_path + '/' + file)


# 获取固定目录下的所有文件名
def all_path(dirname):
    result = []
    # print(dirname)
    for maindir, subdir, file_name_list in os.walk(dirname):
        file_name_list.sort(key=lambda x: int(x[:-4]))
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if platform.system() == 'Windows':
                if apath.split('\\')[1] == '0.csv':
                    continue
                result.append(apath.split('\\')[1])
            else:
                if apath.split('/')[-1] == '0.csv':
                    continue
                result.append(apath.split('/')[-1])

    return result  # 返回文件列表


################  动作识别代码  ###############################
# 客户端启动动作识别，即加载配置文件，配置全局变量
@app.route('/action/init', methods=['POST'])
def InitConifgWithJsonFile():
    configType = request.json['configType']
    if configType == 'init':
        # 初始化配置，默认配置
        configData = loadJsonWithConfig('src/initConfig')
    elif configType == 'reload':
        # 重置用户配置
        configData = loadJsonWithConfig('src/reloadConfig')

    # 定义动作特征降维的维度
    global ACTION_COMPONENTS
    ACTION_COMPONENTS = int(configData['ACTION_COMPONENTS'])
    # HMM 预测得分阈值
    global HMM_SCORE_MAX
    HMM_SCORE_MAX = int(configData['HMM_SCORE_MAX'])
    # 提交第一层数组长度
    global FIRST_ACTION_DATA_WINDOW_SIZE
    FIRST_ACTION_DATA_WINDOW_SIZE = int(configData['FIRST_ACTION_DATA_WINDOW_SIZE'])

    # 第三层数组长度（需要识别的动作数组）
    global ACTION_WINDOW_SIZE
    ACTION_WINDOW_SIZE = int(configData['ACTION_WINDOW_SIZE'])
    # 定义第二层窗口启动长度
    global SECOND_ACTION_DATA_BEGIN_SIZE
    SECOND_ACTION_DATA_BEGIN_SIZE = int(configData['SECOND_ACTION_DATA_BEGIN_SIZE'])
    # 定义第一层数组窗口的滑动的距离
    global MOVIE_SIZE
    MOVIE_SIZE = int(configData['MOVIE_SIZE'])
    global ACTION_DETECT_ACTION_BEGIN_SIZE
    # 定义探测数组的长度（是否开始运动）
    ACTION_DETECT_ACTION_BEGIN_SIZE = int(configData['ACTION_DETECT_ACTION_SIZE'])
    # 定义动作开始斜率
    global ACTION_DETECT_K_START_VALUE
    ACTION_DETECT_K_START_VALUE = int(configData['ACTION_DETECT_K_START_VALUE'])
    # 定义动作结束斜率
    global ACTION_DETECT_K_END_VALUE
    ACTION_DETECT_K_END_VALUE = int(configData['ACTION_DETECT_K_END_VALUE'])

    # 修改reload文件
    if configType == 'init':
        newConfigData = {'ACTION_COMPONENTS': ACTION_COMPONENTS, 'HMM_SCORE_MAX': HMM_SCORE_MAX,
                         'FIRST_ACTION_DATA_WINDOW_SIZE': FIRST_ACTION_DATA_WINDOW_SIZE,
                         'MOVIE_SIZE': MOVIE_SIZE,
                         'SECOND_ACTION_DATA_BEGIN_SIZE': SECOND_ACTION_DATA_BEGIN_SIZE,
                         'ACTION_WINDOW_SIZE': ACTION_WINDOW_SIZE,
                         'ACTION_DETECT_ACTION_SIZE': ACTION_DETECT_ACTION_BEGIN_SIZE,
                         'ACTION_DETECT_K_START_VALUE': ACTION_DETECT_K_START_VALUE,
                         'ACTION_DETECT_K_END_VALUE': ACTION_DETECT_K_END_VALUE
                         }
        dumpJsonWithConfig(newConfigData)
    configData["message"] = 'success {} server config'.format(configType)
    return json.dumps(configData, ensure_ascii=False)


# 接受客户端提交配置文件，并且重置全局变量
@app.route('/action/resetconfig', methods=['POST'])
def ReConifgWithJsonFile():
    # 定义动作特征降维的维度
    global ACTION_COMPONENTS
    ACTION_COMPONENTS = int(request.json['ACTION_COMPONENTS'])
    # HMM 预测得分阈值
    global HMM_SCORE_MAX
    HMM_SCORE_MAX = int(request.json['HMM_SCORE_MAX'])
    # 提交第一层数组长度
    global FIRST_ACTION_DATA_WINDOW_SIZE
    FIRST_ACTION_DATA_WINDOW_SIZE = int(request.json['FIRST_ACTION_DATA_WINDOW_SIZE'])

    # 第三层数组长度（需要识别的动作数组）
    global ACTION_WINDOW_SIZE
    ACTION_WINDOW_SIZE = int(request.json['ACTION_WINDOW_SIZE'])
    # 定义第二层窗口启动长度
    global SECOND_ACTION_DATA_BEGIN_SIZE
    SECOND_ACTION_DATA_BEGIN_SIZE = int(request.json['SECOND_ACTION_DATA_BEGIN_SIZE'])
    # 定义第一层数组窗口的滑动的距离
    global MOVIE_SIZE
    MOVIE_SIZE = int(request.json['MOVIE_SIZE'])
    # 定义探测数组的长度（是否开始运动）
    global ACTION_DETECT_ACTION_BEGIN_SIZE
    ACTION_DETECT_ACTION_BEGIN_SIZE = int(request.json['ACTION_DETECT_ACTION_SIZE'])
    # 定义动作开始斜率
    global ACTION_DETECT_K_START_VALUE
    ACTION_DETECT_K_START_VALUE = int(request.json['ACTION_DETECT_K_START_VALUE'])
    # 定义动作结束斜率
    global ACTION_DETECT_K_END_VALUE
    ACTION_DETECT_K_END_VALUE = int(request.json['ACTION_DETECT_K_END_VALUE'])

    configData = {'ACTION_COMPONENTS': ACTION_COMPONENTS, 'HMM_SCORE_MAX': HMM_SCORE_MAX,
                  'FIRST_ACTION_DATA_WINDOW_SIZE': FIRST_ACTION_DATA_WINDOW_SIZE,
                  'MOVIE_SIZE': MOVIE_SIZE,
                  'SECOND_ACTION_DATA_BEGIN_SIZE': SECOND_ACTION_DATA_BEGIN_SIZE,
                  'ACTION_WINDOW_SIZE': ACTION_WINDOW_SIZE,
                  'ACTION_DETECT_ACTION_SIZE': ACTION_DETECT_ACTION_BEGIN_SIZE,
                  'ACTION_DETECT_K_START_VALUE': ACTION_DETECT_K_START_VALUE,
                  'ACTION_DETECT_K_END_VALUE': ACTION_DETECT_K_END_VALUE}
    # print(configData)
    dumpJsonWithConfig(configData)
    return 'server reset conifg success'


# 保存数据到JSON文件
def dumpJsonWithConfig(data):
    with open("src/reloadConfig", 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False)


# 从配置文件加载JSON数据
def loadJsonWithConfig(fileName):
    with open(fileName) as json_file:
        data = json.load(json_file)
        return data

# 设置识别模型
# http://localhost:5001/action/recognition/1
@app.route('/action/recognition/<string:model>')
def set_recogintion_model(model):
    return f'success set model is hmm'

@app.route('/action/getstring')
def get_string():
    return 'get success'


@app.route('/user/json', methods=['POST'])
def get_json():
    # print(request.headers)
    print(request.json)
    # print(request.json['nodeCount'])
    # 返回json
    return Response(json.dumps('post string'), mimetype='application/json')


# 动作识别 (测试1)
@app.route('/action/poststring', methods=['POST'])
def post_string():
    # print('do post')
    # print(request.headers)
    print(request.json)
    # print(request.json['nodeCount'])
    # print(request.json['sensorData'])
    '''
    sensordata.nodeCount = request.json['nodeCount']
    sensordata.sensorData = request.json['sensorData']
    print(sensordata.nodeCount + ':' + sensordata)
    '''
    return 'None Action'


# 历史动作识别 (测试2)
@app.route('/action/history_action', methods=['GET'])
def history_action():
    try:
        dataMat = pd.read_csv('src/test/data2.csv', names=O_COLUMNS).reset_index().ix[:, 1:]
        count = 0
        action_data_window = []

        while count < len(dataMat):
            # 模拟数据一帧 一帧的读取
            data = dataMat.loc[count]
            action_data_window.append([c for c in np.array(data)])
            if len(action_data_window) == FIRST_ACTION_DATA_WINDOW_SIZE:
                action_data_window = DataFrame(action_data_window, columns=O_COLUMNS)

                # 获取动作类别
                action_clas = dealwithdynamicdata(action_data_window, FIRST_ACTION_DATA_WINDOW_SIZE, ACTION_COMPONENTS)
                print(action_clas)
                action_data_window = []
            count += 1
        return 'history action over'
    except BaseException  as be:
        print(be)
        return 'have error 500'


# 为实时识别，以当前时间创建文件，保存数据
@app.route('/action/addfile', methods=['GET'])
def addActionDataFile():
    import os, sys, time
    strTime = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '.csv'
    global ACTION_DATA_FILE
    ACTION_DATA_FILE = 'static/data/actionDataRecognition/' + strTime
    # open(ACTION_DATA_FILE, 'w')
    return 'create file {}'.format(ACTION_DATA_FILE)


# 动作识别（测试，第一版本 已经停用）
@app.route('/action/recognition', methods=['POST'])
def action_recognition():
    try:
        # print(request.json)
        # sensorData 含有60帧数据以@分割，一帧数据由7个节点以；分割，一个节点由9轴数据构成以，分割
        dataMat = []
        allDataString = request.json['sensorData'].split('@')  # 按帧数分

        for dataString in allDataString:  # 60帧
            if dataString == '': continue
            sevenDataAndSeq = []
            dataNode = dataString.split(';')  # 按节点分
            for subDataNode in dataNode:  # 7个节点
                dataArrayAndSeq = subDataNode.split(',')  #
                dataNumArrayAndSeq = [np.float(c) for c in dataArrayAndSeq]
                sevenDataAndSeq = sevenDataAndSeq + dataNumArrayAndSeq[1:]

            dataMat.append(sevenDataAndSeq)
        dataMat = pd.DataFrame(dataMat, columns=O_COLUMNS)
        # print(dataMat)
        # dataMat.to_csv('src/test/data.csv', mode='a', header=False)

        action_classes = dealwithdynamicdata(dataMat, FIRST_ACTION_DATA_WINDOW_SIZE, ACTION_COMPONENTS)
        if action_classes == '1':
            action_classes = 'action1:正手攻球'
        elif action_classes == '2':
            action_classes = 'action2:正手拉球'
        elif action_classes == '3':
            action_classes = 'action3:正手搓球'
        elif action_classes == '4':
            action_classes = 'action4:正手挑球'
        elif action_classes == '5':
            action_classes = 'action5:反手拨球'
        elif action_classes == '6':
            action_classes = 'action6:反手拉球'
        elif action_classes == '7':
            action_classes = 'action7:反手搓球'
        elif action_classes == '8':
            action_classes = 'action8:反手拧球'
        else:
            action_classes = 'None Action'

        print(action_classes)
        return action_classes
    except BaseException as be:
        print(be)
        return 'have an error 500'
    # return 'None Action'


#  实时动作识别
@app.route('/action/greatRecognition', methods=['POST'])
def action_greatRecognition():
    try:
        dataMat = []
        allDataString = request.json['sensorData'].split('@')  # 按帧数分
        # print(allDataString)

        for dataString in allDataString:  # 60帧
            if dataString == '': continue
            sevenDataAndSeq = []
            dataNode = dataString.split(';')  # 按节点分
            for subDataNode in dataNode:  # 7个节点
                dataArrayAndSeq = subDataNode.split(',')  #
                dataNumArrayAndSeq = [np.float(c) for c in dataArrayAndSeq]
                sevenDataAndSeq = sevenDataAndSeq + dataNumArrayAndSeq[1:]

            dataMat.append(sevenDataAndSeq)
        dataMat = pd.DataFrame(dataMat, columns=O_COLUMNS)

        action_classes, action_score = GreatDealWithDynamicData(dataMat)
        action_score = str(action_score)
        if action_classes == '1':
            action_classes = 'action1:正手快攻 , 得分：' + action_score
        elif action_classes == '2':
            action_classes = 'action2:快拨 , 得分：' + action_score
        elif action_classes == '3':
            action_classes = 'action3:加转弧圈 , 得分：' + action_score
        elif action_classes == '4':
            action_classes = 'action4:前冲弧圈 , 得分：' + action_score
        elif action_classes == '5':
            action_classes = 'action5:反手搓球 , 得分：' + action_score
        elif action_classes == '6':
            action_classes = 'action6:正手快攻 , 得分：' + action_score
        elif action_classes == '7':
            action_classes = 'action7:快推 , 得分：' + action_score
        elif action_classes == '8':
            action_classes = 'action8:加转弧圈 , 得分：' + action_score
        elif action_classes == '9':
            action_classes = 'action9:前冲弧圈 , 得分：' + action_score
        elif action_classes == '10':
            action_classes = 'action10:反手搓球 , 得分：' + action_score
        else:
            action_classes = 'None Action'
        print('动作类别{}'.format(action_classes))
        return action_classes
    except BaseException as be:

        print("实时动作识别出现异常{}".format(traceback.format_exc()))
        return 'have an error 500'


######################################## 动作识别  begin #########################################

def GreatDealWithDynamicData(dataMat):
    action_classes = 'None Action'
    action_score = -100
    count = 0
    global action_data_window_queue
    global second_action_data_queue
    global isActionDetectStart
    global isActionDetectEnd
    global actionDetectEndCounter

    # begin保存数据(可删除）
    # testDataMat = DataFrame(columns=O_COLUMNS).append(dataMat, ignore_index=True)
    # testDataMat['ACC'] = testDataMat.apply(lambda row:
    #                                       (float(row['bAX']) + float(row['bAY']) + float(row['bAZ'])), axis=1)
    # testDataMat.to_csv(ACTION_DATA_FILE, mode='a', header=False)
    # end 保存数据

    # print(dataMat)
    while count < len(dataMat):
        # 第一层的长度
        if len(action_data_window_queue) == FIRST_ACTION_DATA_WINDOW_SIZE:
            # 窗口移动
            action_data_window_queue = action_data_window_queue[MOVIE_SIZE:]
            # 模拟数据一帧 一帧的读取
        data = dataMat.loc[count]

        # data['ACC'] = float(data['bAX']) + float(data['bAY']) + float(data['bAZ'])
        data['ACC'] = math.sqrt((data['bAX']) ** 2 + float(data['bAY']) ** 2 + float(data['bAZ']) ** 2)

        action_data_window_queue = action_data_window_queue.append(data, ignore_index=True)
        # 获取探测数组
        action_slope_queue = action_data_window_queue['ACC'][ACTION_DETECT_ACTION_BEGIN_SIZE:].values

        # 判断动作是否开始
        if (not isActionDetectStart) and isRegressionArray(action_slope_queue, mode='start'):
            # print('检测到动作开始')
            isActionDetectStart = True
            second_action_data_queue = action_data_window_queue[-SECOND_ACTION_DATA_BEGIN_SIZE:]

        # 重新开始
        if len(second_action_data_queue) > FIRST_ACTION_DATA_WINDOW_SIZE:
            isActionDetectStart = False
            isActionDetectEnd = False

        # 如果动作开始，开始提取动作窗口
        if isActionDetectStart:
            second_action_data_queue = second_action_data_queue.append(data, ignore_index=True)

            # 判断动作是否结束
            if (not isActionDetectEnd) and isRegressionArray(action_slope_queue, mode='end'):
                # print('检测到动作结束')
                isActionDetectEnd = True
                actionDetectEndCounter = 0  # 启动计数器

            # 动作结束
            if isActionDetectEnd:
                actionDetectEndCounter += 1
                # 此时认为动作结束（结束之后，再添加20个长度）
                if actionDetectEndCounter > SECOND_ACTION_DATA_BEGIN_SIZE:
                    isActionDetectStart = False  # 动作提取完毕
                    isActionDetectEnd = False

                    # print('--第二层动作长度：' + str(len(second_action_data_queue)))
                    # array_acc = second_action_data_queue['ACC'].values
                    max_index = second_action_data_queue['ACC'].idxmax()
                    # print(str(len(second_action_data_queue)) + ":" + str(max_index))
                    # 动作窗口 action_window
                    action_window = second_action_data_queue[
                                    max_index - int(ACTION_WINDOW_SIZE * 0.5):max_index + int(ACTION_WINDOW_SIZE * 0.5)]
                    if len(action_window) == ACTION_WINDOW_SIZE:
                        # 开始hmm 识别
                        new_action_windows = DataFrame(columns=O_COLUMNS).append(action_window.drop(['ACC'], axis=1),
                                                                                 ignore_index=True)
                        # print(new_action_windows)
                        # 特征提取
                        X = Feature_process(new_action_windows)
                        # 重组数据
                        X = complex_data.append(X, ignore_index=True)
                        # 数据归一化
                        X = preprocessing.normalize(X)
                        X = DataFrame(X, columns=N_COLUMNS)
                        # 特征降维
                        X_de = decomposition(X, de_str='PCA', n_components=ACTION_COMPONENTS).round(6)
                        # 开始 hmm 分类
                        seen = np.array(X_de.round(6))[-1, :]
                        action_number, score_max = predictByGussianHMM(seen, ACTION_COMPONENTS)
                        action_score = assessScoreSystem(score_max)
                        if not action_number == -1:
                            action_classes = str(action_number)
                            # print('识别出动作类别是：{0},得分是：{1}'.format(str(action_number), score_max))
                            # print(X_de)
                            return (action_classes, action_score)
        # end if(count 循环计数器）
        count += 1
    return (action_classes, action_score)
    # end while


# 旧的实时识别，处理函数 非greate
def dealwithdynamicdata(dataMat, DATA_WINDOWS_SIZE, COMPONENTS):
    action_classes = 'None Action'
    # 复合数据
    # print(dataMat)
    if len(dataMat) == DATA_WINDOWS_SIZE:
        # print(dataMat)
        # 特征提取
        X = Feature_process(dataMat)
        # 重组数据
        X = complex_data.append(X, ignore_index=True)

        # 数据归一化
        X = preprocessing.normalize(X)
        X = DataFrame(X, columns=N_COLUMNS)

        # 特征降维
        X_de = decomposition(X, de_str='PCA', n_components=ACTION_COMPONENTS).round(6)
        # X_de.to_csv('src/test/data_de.csv', mode='a', header=False)

        # 开始 hmm 分类
        seen = np.array(X_de.round(6))[-1, :]
        action_number, score_max = predictByGussianHMM(seen, ACTION_COMPONENTS)
        action_score = assessScoreSystem(score_max)
        if not action_number == -1:
            # print('识别出动作类别是：{0},得分是：{1}'.format(str(action_number), action_score))
            action_classes = str(action_number)
            return action_classes
        # print(X_de)
    return action_classes


# 评分规则 （0，15）->（60，90）
def assessScoreSystem(score):
    score = abs(float(score))
    if score <= 15:
        score = score * 6
    elif (score > 15) and (score < 20):
        score = 90 + (score - 15) * 2
    else:
        score = 100
    return round(score, 2)


# 通过线性回归，判断斜率
def isRegressionArray(array, mode='start'):
    '''
    :param array: 探测数组
    :param mode: 探测动作开始start 或者结束end
    :return:
    '''
    array = np.array(array)
    reg = linear_model.LinearRegression()
    # 对应序号是 range(len(data))
    reg.fit(np.array(range(len(array))).reshape(-1, 1), np.array(array).reshape(-1, 1))
    # 斜率为
    # print(mode + '斜率是:' + str(reg.coef_[0][0]))
    # 截距为
    # print(reg.intercept_)
    if mode == 'start':
        # print('开始：array {} ,斜率 {}'.format(array, reg.coef_[0][0]))
        k_value = ACTION_DETECT_K_START_VALUE
        if reg.coef_[0][0] > k_value:
            return True
        else:
            return False
    elif mode == 'end':
        # print('结束：array {} ,斜率 {}'.format(array, reg.coef_[0][0]))
        k_value = ACTION_DETECT_K_END_VALUE
        if (reg.coef_[0][0] < 0) and (abs(reg.coef_[0][0]) > abs(k_value)):
            return True
        else:
            return False

# 斜率判断，返回动作是否开始
def isSlopeArray(array, mode='start'):
    array = np.array(array)
    array_len = len(array)
    if array_len != np.abs(ACTION_DETECT_ACTION_BEGIN_SIZE):
        return False

    else:
        if mode == 'start':
            for index in range(array_len):
                if index == array_len - 1:
                    k_value = array[index] - array[0]
                else:
                    k_value = array[index + 1] - array[index]
                if k_value < ACTION_DETECT_K_START_VALUE:
                    return False
            return True
        elif mode == 'end':
            for index in range(array_len):
                if index == array_len - 1:
                    k_value = array[index] - array[0]
                else:
                    k_value = array[index + 1] - array[index]
                if k_value > ACTION_DETECT_K_END_VALUE:
                    return False
            return True


# 快速傅里叶变换
def fft_T_function(dataMat):
    '''
    axis = 0 垂直 做变换
    :param dataMat:
    :return:
    '''
    dataMat = dataMat.T
    dataF = dataMat.apply(np.fft.fft)
    data = dataF.apply(lambda x: np.abs(x.real), axis=1)

    df = []
    for array in data:
        df.append(np.max(array))
    df = np.array(df).reshape(1, 7 * 9)  # 7个节点*9个轴数据
    data = DataFrame(df, columns=O_COLUMNS)
    data = Series(np.array(data)[0], index=O_COLUMNS)
    return data


# end 快速傅里叶变换

# 特征处理
def Feature_process(dataMat):
    '''
    将数据集转换为特征矩阵，并标记标签。12一组，60/12=5 ，5个状态
    （一个动作由5维数据表示） 45* 7 = 315 个特征值
    A 均值，C 协方差，K 峰度，S 偏度， F 快速傅里叶（FFT值）
    :param dataMat: 数据集
    '''

    X = DataFrame(columns=N_COLUMNS)
    # 均值 A,协方差C，峰值K,偏度S，
    dataA = dataMat.apply(np.average)
    dataC = dataMat.apply(np.cov)
    # 分别使用df.kurt()方法和df.skew()即可完成峰度he偏度计算
    dataK = dataMat.kurt()
    dataS = dataMat.skew()
    # 使用fft函数对余弦波信号进行傅里叶变换。并取绝对值
    dataF = fft_T_function(dataMat)

    df = DataFrame(
        [[dataA.aAX, dataA.aAY, dataA.aAZ, dataC.aAX, dataC.aAY, dataC.aAZ, dataK.aAX, dataK.aAY, dataK.aAZ,
          dataS.aAX, dataS.aAY, dataS.aAZ, dataF.aAX, dataF.aAY, dataF.aAZ,
          dataA.aWX, dataA.aWY, dataA.aWZ, dataC.aWX, dataC.aWY, dataC.aWZ, dataK.aWX, dataK.aWY, dataK.aWZ,
          dataS.aWX, dataS.aWY, dataS.aWZ, dataF.aWX, dataF.aWY, dataF.aWZ,
          dataA.aHX, dataA.aHY, dataA.aHZ, dataC.aHX, dataC.aHY, dataC.aHZ, dataK.aHX, dataK.aHY, dataK.aHZ,
          dataS.aHX, dataS.aHY, dataS.aHZ, dataF.aHX, dataF.aHY, dataF.aHZ,

          dataA.bAX, dataA.bAY, dataA.bAZ, dataC.bAX, dataC.bAY, dataC.bAZ, dataK.bAX, dataK.bAY, dataK.bAZ,
          dataS.bAX, dataS.bAY, dataS.bAZ, dataF.bAX, dataF.bAY, dataF.bAZ,
          dataA.bWX, dataA.bWY, dataA.bWZ, dataC.bWX, dataC.bWY, dataC.bWZ, dataK.bWX, dataK.bWY, dataK.bWZ,
          dataS.bWX, dataS.bWY, dataS.bWZ, dataF.bWX, dataF.bWY, dataF.bWZ,
          dataA.bHX, dataA.bHY, dataA.bHZ, dataC.bHX, dataC.bHY, dataC.bHZ, dataK.bHX, dataK.bHY, dataK.bHZ,
          dataS.bHX, dataS.bHY, dataS.bHZ, dataF.bHX, dataF.bHY, dataF.bHZ,

          dataA.cAX, dataA.cAY, dataA.cAZ, dataC.cAX, dataC.cAY, dataC.cAZ, dataK.cAX, dataK.cAY, dataK.cAZ,
          dataS.cAX, dataS.cAY, dataS.cAZ, dataF.cAX, dataF.cAY, dataF.cAZ,
          dataA.cWX, dataA.cWY, dataA.cWZ, dataC.cWX, dataC.cWY, dataC.cWZ, dataK.cWX, dataK.cWY, dataK.cWZ,
          dataS.cWX, dataS.cWY, dataS.cWZ, dataF.cWX, dataF.cWY, dataF.cWZ,
          dataA.cHX, dataA.cHY, dataA.cHZ, dataC.cHX, dataC.cHY, dataC.cHZ, dataK.cHX, dataK.cHY, dataK.cHZ,
          dataS.cHX, dataS.cHY, dataS.cHZ, dataF.cHX, dataF.cHY, dataF.cHZ,

          dataA.dAX, dataA.dAY, dataA.dAZ, dataC.dAX, dataC.dAY, dataC.dAZ, dataK.dAX, dataK.dAY, dataK.dAZ,
          dataS.dAX, dataS.dAY, dataS.dAZ, dataF.dAX, dataF.dAY, dataF.dAZ,
          dataA.dWX, dataA.dWY, dataA.dWZ, dataC.dWX, dataC.dWY, dataC.dWZ, dataK.dWX, dataK.dWY, dataK.dWZ,
          dataS.dWX, dataS.dWY, dataS.dWZ, dataF.dWX, dataF.dWY, dataF.dWZ,
          dataA.dHX, dataA.dHY, dataA.dHZ, dataC.dHX, dataC.dHY, dataC.dHZ, dataK.dHX, dataK.dHY, dataK.dHZ,
          dataS.dHX, dataS.dHY, dataS.dHZ, dataF.dHX, dataF.dHY, dataF.dHZ,

          dataA.eAX, dataA.eAY, dataA.eAZ, dataC.eAX, dataC.eAY, dataC.eAZ, dataK.eAX, dataK.eAY, dataK.eAZ,
          dataS.eAX, dataS.eAY, dataS.eAZ, dataF.eAX, dataF.eAY, dataF.eAZ,
          dataA.eWX, dataA.eWY, dataA.eWZ, dataC.eWX, dataC.eWY, dataC.eWZ, dataK.eWX, dataK.eWY, dataK.eWZ,
          dataS.eWX, dataS.eWY, dataS.eWZ, dataF.eWX, dataF.eWY, dataF.eWZ,
          dataA.eHX, dataA.eHY, dataA.eHZ, dataC.eHX, dataC.eHY, dataC.eHZ, dataK.eHX, dataK.eHY, dataK.eHZ,
          dataS.eHX, dataS.eHY, dataS.eHZ, dataF.eHX, dataF.eHY, dataF.eHZ,

          dataA.fAX, dataA.fAY, dataA.fAZ, dataC.fAX, dataC.fAY, dataC.fAZ, dataK.fAX, dataK.fAY, dataK.fAZ,
          dataS.fAX, dataS.fAY, dataS.fAZ, dataF.fAX, dataF.fAY, dataF.fAZ,
          dataA.fWX, dataA.fWY, dataA.fWZ, dataC.fWX, dataC.fWY, dataC.fWZ, dataK.fWX, dataK.fWY, dataK.fWZ,
          dataS.fWX, dataS.fWY, dataS.fWZ, dataF.fWX, dataF.fWY, dataF.fWZ,
          dataA.fHX, dataA.fHY, dataA.fHZ, dataC.fHX, dataC.fHY, dataC.fHZ, dataK.fHX, dataK.fHY, dataK.fHZ,
          dataS.fHX, dataS.fHY, dataS.fHZ, dataF.fHX, dataF.fHY, dataF.fHZ,

          dataA.gAX, dataA.gAY, dataA.gAZ, dataC.gAX, dataC.gAY, dataC.gAZ, dataK.gAX, dataK.gAY, dataK.gAZ,
          dataS.gAX, dataS.gAY, dataS.gAZ, dataF.gAX, dataF.gAY, dataF.gAZ,
          dataA.gWX, dataA.gWY, dataA.gWZ, dataC.gWX, dataC.gWY, dataC.gWZ, dataK.gWX, dataK.gWY, dataK.gWZ,
          dataS.gWX, dataS.gWY, dataS.gWZ, dataF.gWX, dataF.gWY, dataF.gWZ,
          dataA.gHX, dataA.gHY, dataA.gHZ, dataC.gHX, dataC.gHY, dataC.gHZ, dataK.gHX, dataK.gHY, dataK.gHZ,
          dataS.gHX, dataS.gHY, dataS.gHZ, dataF.gHX, dataF.gHY, dataF.gHZ,
          ]], columns=N_COLUMNS)

    return df
    # 特征处理 end


# 数据降维
def decomposition(X, de_str='PCA', n_components=10):
    '''
        对实验数据降维
    :param X:数据集
    :return:X_pca
    '''

    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD

    if de_str == 'PCA':
        de_model = PCA(n_components=n_components)
    else:
        de_model = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)

    # X = DataFrame().append(X, ignore_index=True)

    de_model.fit(X)
    X_de = de_model.transform(X)
    X_de = DataFrame(X_de, columns=['pca' + str(i) for i in np.arange(n_components)])

    # 返回各个成分各自的方差百分比(贡献率)
    # print('成分各自的方差百分比(贡献率):{}'.format(np.add.reduce(de_model.explained_variance_ratio_)))
    # print(de_model.explained_variance_ratio_)

    # columns = ['pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'SPE']

    # print('特征降维处理完毕...')
    return X_de


# 隐马尔可夫预测：判断动作类型
def predictByGussianHMM(seen, n_components):
    '''
    :param seen:
    :param n_components: 数据维度
    :return: 返回 动作编号和动作得分
    '''
    seen = np.array(seen).reshape(-1, n_components)
    # hmm_suffix_array = ['1.pkl', '2.pkl', '3.pkl', '4.pkl', '5.pkl', '6.pkl', '7.pkl', '8.pkl']
    hmm_suffix_array = ['1.pkl', '2.pkl', '3.pkl', '4.pkl', '5.pkl']
    # print(seen)
    score_list = []
    for hmm_suffix in hmm_suffix_array:
        my_file = "src/hmm_model/hmm_GaussianHMM" + hmm_suffix
        hmm_guss_model = joblib.load(my_file)
        score = hmm_guss_model.score(seen)
        score_list.append(score)

    # print(score_list)
    # 打印最大值
    score_max = np.max(score_list)

    if score_max < HMM_SCORE_MAX:
        return (-1, score_max)

    # 返回最大值对应的动作
    action_number = np.argmax(score_list) + 1
    # print('第二次得分最大值：' + str(np.max(score_list)))
    return (action_number, score_max)
    # print('end................')


# ######################################## 动作识别   end  #########################################
if __name__ == '__main__':
    # app.config['JSON_AS_ASCII'] = False
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
