# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     test_2
   Description:   ...
   Author:        cqh
   date:          2022/10/28 15:31
-------------------------------------------------
   Change Activity:
                  2022/10/28:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import time
import psutil

print("pid", os.getpid())

mem = psutil.virtual_memory()
# 系统总计内存
zj = float(mem.total) / 1024 / 1024 / 1024
# 系统已经使用内存
ysy = float(mem.used) / 1024 / 1024 / 1024

# 系统空闲内存
kx = float(mem.free) / 1024 / 1024 / 1024

memory_capacity = zj * 0.25

print('系统总计内存:%d.3GB' % zj)
print('系统已经使用内存:%d.3GB' % ysy)
print('系统空闲内存:%d.3GB' % kx)
print("至少需要留有 {:.3f} GB的空间".format(memory_capacity))