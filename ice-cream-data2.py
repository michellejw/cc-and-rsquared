#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create an example dataset

Created on Thu Feb 14 13:00:11 2019

@author: michw
"""

import pandas as pd

data_url = "https://michw.com/DATA/precip_temp_data.xlsx"

# Read ice cream data into a data frame
df = pd.read_excel(data_url)

# Make column names easier to work with
df.columns = ['Year','Month','dayOfMonth','JulDay','NormalMaxTemp',\
              'NormalMinTemp','MeanTemp','NormalPrecip']

# Create a new temperature vector that is 

