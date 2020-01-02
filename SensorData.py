#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2018/11/19 20:08
Describe：
    
    
"""


class SensorData:

    __slots__ = ('__nodeCount', '__sensorData')  # 用tuple定义允许绑定的属性名称

    def __init__(self, nodeCount, sensorData):
        self.__nodeCount = nodeCount
        self.__sensorData = sensorData

    @property
    def nodeCount(self):
        return self.__nodeCount

    @nodeCount.setter
    def nodeCount(self, nodeCount):
        self.__nodeCount = nodeCount

    @property
    def sensorData(self):
        return self.__sensorData

    @sensorData.setter
    def sensorData(self, sensorData):
        self.__sensorData = sensorData

    def __str__(self):
        data = self.__sensorData.split(",")
        return data[1]+','+data[2]+','+data[3]+','+data[4]+','+data[5]+','+data[6]+','+data[7]+','+data[8]+','+data[9]
