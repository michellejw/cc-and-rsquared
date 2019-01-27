#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:11:49 2019
Generate Rita's sample ice cream data 

@author: michw
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# import matplotlib
# %matplotlib qt

#%%
fsize = 14

npts = 40
sp.random.seed(seed=20190114) # Make it the same every time!
x = np.random.uniform(low=25,high=105,size=npts) 
err = np.random.normal(loc=0.0, scale = 60.0,size=npts)

b = -84.875
m = 3.875

# Generate data that includes an error term
y = abs((m*x) + b + err)
## Adjust the highest values a bit to make sales flatten at high temps
y[33] = 205
y[10] = 55
#for ydex in np.arange(len(y)):
#    if y[ydex] > 250:
#        y[ydex] = y[ydex]-100

# Fit a line to the data
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
yfit = (slope*x) + intercept

plt.figure(1)
plt.clf()
plt.plot(x,abs(y),'.')
plt.xlabel('Temperature (degrees F)',fontsize=fsize)
plt.ylabel('Sales (# cones)',fontsize=fsize)
plt.show()

#plt.savefig('salesVStemp_noline.png', transparent=True)

plt.figure(2)
plt.clf()
plt.plot(x,abs(y),'.')
plt.plot(x,yfit,'-')
plt.xlabel('Temperature (degrees F)',fontsize=fsize)
plt.ylabel('Sales (# cones)',fontsize=fsize)
plt.show()

#plt.savefig('salesVStemp_linefit.png', transparent=True)

plt.figure(3)
plt.clf()
plt.plot(x,yfit,'-',color='#ff7f0e')
plt.plot(x,y,'.',color='#1f77b4')
plt.vlines(x,yfit,y,color=(0.7, 0.7, 0.7))
plt.xlabel('Temperature (degrees F)',fontsize=fsize)
plt.ylabel('Sales (# cones)',fontsize=fsize)
plt.show()

#plt.savefig('salesVStemp_residuals.png',transparent=True)

# Save data
df = pd.DataFrame({'Temperature': x, 'nCones': y})
#df.to_csv('icecreamdata.csv',index=False,float_format = '%i')

#%%
'''
Examples of Different correlation coefficient values
'''

sp.random.seed(seed=5) # Make it the same every time!
x = np.random.uniform(low=-1,high=1,size=50) 
err = np.random.normal(loc=0.0, scale = 0.3,size=50)
err2 = np.random.normal(loc=0.0, scale = 0.25, size = 50)


def removeaxes(ax):
    for axitem in ax:
        axitem.get_yaxis().set_visible(False)
        axitem.get_xaxis().set_visible(False)
#        axitem.spines['top'].set_visible(False)
#        axitem.spines['right'].set_visible(False)
#        axitem.spines['left'].set_visible(False)
#        axitem.spines['bottom'].set_visible(False)

def setaxlims(ax,lower,upper):
    for axitem in ax:
        axitem.set_ylim(lower,upper)
        axitem.set_xlim(lower,upper)
    

plt.figure(figsize=(9,5))
grid = plt.GridSpec(2,3, hspace=0, wspace=0)
ax1 = plt.subplot(grid[0, 0])
ax2 = plt.subplot(grid[0, 1])
ax3 = plt.subplot(grid[0, 2])
ax4 = plt.subplot(grid[1, 0])
ax5 = plt.subplot(grid[1, 1])
ax6 = plt.subplot(grid[1, 2])
removeaxes([ax1,ax2,ax3,ax4,ax5,ax6])
setaxlims([ax1,ax2,ax3,ax4,ax5,ax6],-1.5,1.5)

# CC = -1
y = -x
ax1.plot(x,y,'.')

# CC = 0
y = np.zeros(len(x))
ax2.plot(x,y,'.')

# CC = 1
y = x
ax3.plot(x,y,'.')

# CC = -0.?
y = -x-err
ax4.plot(x,y,'.')

# CC = 0
y = err2
ax5.plot(x,y,'.')

# CC = 0.?
y = x+err
ax6.plot(x,y,'.')

#plt.savefig('SampleCCs.png', transparent=True, dpi=300)

#%%
'''
Examples of different R^2 values (coefficient of determination)

'''
sp.random.seed(seed=55) # Make it the same every time!
npoints = 25
x = np.random.uniform(low=0,high=1,size=npoints) 

# Set up axes
plt.figure(num=5,figsize=(9,3))
plt.clf()
grid = plt.GridSpec(1,3, hspace=0, wspace=0)
ax1 = plt.subplot(grid[0, 0])
ax2 = plt.subplot(grid[0, 1])
ax3 = plt.subplot(grid[0, 2])
removeaxes([ax1,ax2,ax3])
setaxlims([ax1,ax2,ax3],-0.3,1.3)


# R2
m = 1
b = 0
err1 = np.random.normal(loc=0.0, scale = .55,size=npoints)
#err1 = np.random.uniform(low=-1,high=1,size=npoints) 
y_mod1 = (m*x) + b
y_err1 = y_mod1 + err1
# Fit a line to the data
slope1, intercept1, r_value1, p_value, std_err = stats.linregress(x,y_err1)
y_fit1 = (slope1*x) + intercept1
ax1.plot(x,y_fit1,'-')
ax1.plot(x,y_err1,'.')

# R2
m = 1
b = 0
err2 = np.random.normal(loc=0.0, scale = 0.2,size=npoints)
y_mod2 = (m*x) + b
y_err2 = y_mod2 + err2
# Fit a line to the data
slope2, intercept2, r_value2, p_value, std_err = stats.linregress(x,y_err2)
y_fit2 = (slope2*x) + intercept2
ax2.plot(x,y_fit2,'-')
ax2.plot(x,y_err2,'.')

# R2
m = 1
b = 0
err3 = np.random.normal(loc=0.0, scale = 0.03,size=npoints)
y_mod3 = (m*x) + b
y_err3 = y_mod3 + err3
# Fit a line to the data
slope3, intercept3, r_value3, p_value, std_err = stats.linregress(x,y_err3)
y_fit3 = (slope3*x) + intercept3
ax3.plot(x,y_fit3,'-')
ax3.plot(x,y_err3,'.')

#plt.savefig('SampleRsquared.png', transparent=True, dpi=300)
