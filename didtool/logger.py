# -*- coding: UTF-8 -*-
"""=================================================
@Project -> File   ：didtool -> logger
@IDE    ：PyCharm
@Author ：Yangfan
@Date   ：2020/6/8 12:39
@Desc   ：
=================================================="""
import sys


class Logger(object):
    def __init__(self, file_name="Default.log"):
        self.terminal = sys.stdout
        self.log = open(file_name, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        # every loop of writing flush into file to prevent unexpected end
        # 每次写入后刷新到文件中，防止程序意外结束
        self.flush()

    def flush(self):
        self.log.flush()
