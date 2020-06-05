# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：didtool -> Logger
@IDE    ：PyCharm
@Author ：Yangfan
@Date   ：2020/6/4 20:56
@Desc   ：
=================================================='''

import sys
class Logger(object):
    """
        Class for the didtool relative logs
        example:
            plus the code ahead your codeing area
            sys.stdout = Logger("./bayes_parameter_search.txt")
    """
    def __init__(self, file_name="Default.log"):
        self.terminal = sys.stdout
        self.log = open(file_name, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # 每次写入后刷新到文件中，防止程序意外结束

    def flush(self):
        self.log.flush()


