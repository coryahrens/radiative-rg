'''
This code demonstrates how to find shock fronts 
and do polynomial fitting on log-log plots.
'''

import numpy as np
import matplotlib.pyplot as plt

def getShockFrontLocation(u0,u1,x):
    # find shock fronts
    gradu0 = np.gradient(u0) # take derivative
    frontIDX0 = np.where(gradu0 == max(gradu0)) # find location of maximum derivative

    gradu1 = np.gradient(u1)
    frontIDX1 = np.where(gradu1 == max(gradu1))

    z0 = x[frontIDX0] # shock front at t0
    z1 = x[frontIDX1] # shock front at t1
    return z0,z1

def getAlpha(t0,u0,t1,u1,x):

    # find shock fronts
    gradu0 = np.gradient(u0) # take derivative
    frontIDX0 = np.where(gradu0 == max(gradu0)) # find location of maximum derivative

    gradu1 = np.gradient(u1)
    frontIDX1 = np.where(gradu1 == max(gradu1))

    z0 = x[frontIDX0] # shock front at t0
    z1 = x[frontIDX1] # shock front at t1

    # polynomial fitting 
    zvals = np.array([z0,z1])
    zvals = np.squeeze(zvals)
    tvals = np.array([t0,t1])

    logz = np.log(zvals)
    logt = np.log(tvals)

    coeffs = np.polyfit(logt,logz,deg=1)

    return coeffs[0]
