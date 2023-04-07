# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     lab_plot_test
   Description:   ...
   Author:        cqh
   date:          2023/3/8 14:58
-------------------------------------------------
   Change Activity:
                  2023/3/8:
-------------------------------------------------
"""
__author__ = 'cqh'
# 查询当前系统所有字体
from matplotlib.font_manager import FontManager
import subprocess

mpl_fonts = set(f.name for f in FontManager().ttflist)

print('all font list get from matplotlib.font_manager:')
for f in sorted(mpl_fonts):
    print('\t' + f)