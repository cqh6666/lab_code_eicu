# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     email_test
   Description:   ...
   Author:        cqh
   date:          2022/9/23 16:53
-------------------------------------------------
   Change Activity:
                  2022/9/23:
-------------------------------------------------
"""
__author__ = 'cqh'

import time

from email_api import send_success_mail

st = time.time()
time.sleep(1)
send_success_mail("test程序", st, time.time())