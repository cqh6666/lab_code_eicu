# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     sqlSelect
   Description:   ...
   Author:        cqh
   date:          2022/9/6 14:07
-------------------------------------------------
   Change Activity:
                  2022/9/6:
-------------------------------------------------
"""
__author__ = 'cqh'

import math

import psycopg2
import pandas as pd
from psycopg2 import extras as ex
import numpy as np
import joblib
from queue import Queue