'''
This code demonstrates how to find shock fronts 
and do polynomial fitting on log-log plots.
'''

import numpy as np
import matplotlib.pyplot as plt

# intrgtation time
b = 0.2 

# read in arrays
u0 = np.loadtxt('u0.txt')
u1 = np.loadtxt('u1.txt')
x  = np.loadtxt('x.txt')

# find shock fronts
gradu0 = np.gradient(u0) # take derivative
frontIDX0 = np.where(gradu0 == max(gradu0)) # find location of maximum derivative

gradu1 = np.gradient(u1)
frontIDX1 = np.where(gradu1 == max(gradu1))

z0 = x[frontIDX0] # shock front at t0
z1 = x[frontIDX1] # shock front at t1

v0 = (z1-z0)/0.2

# polynomial fitting 
zvals = np.array([z0,z1])
zvals = np.squeeze(zvals)
tvals = np.array([1.,1.+b])

logz = np.log(zvals)
logt = np.log(tvals)

coeffs = np.polyfit(logt,logz,deg=1)
poly = np.poly1d(coeffs)

# display result 
print "Polynomial Fit For Position:", poly

# check velocity

v0_pred = coeffs[0]*np.exp(coeffs[1])*1.2**(coeffs[0]-1.)

print "Predicted Velocity:", v0_pred
print "Calculated Velocity:", v0[0]
print "Percent Relative Error:", 100.*abs(v0_pred - v0[0])/v0[0] 

#plot 
plt.figure(1)
plt.plot(x.T,u0,'g',x.T,u1,'r')
plt.legend([r'$t_0 = 1$',r'$t_1 = 1.2$'])
plt.grid()

plt.figure(2)
t = np.linspace(0.01,2,100)
plt.plot(t,np.exp(coeffs[1])*t**(coeffs[0]),'b',tvals,zvals,'b*')
plt.plot(t,coeffs[0]*np.exp(coeffs[1])*t**(coeffs[0]-1.),'r',[1.2],v0,'ro')
plt.legend(['Position','Position Data', 'Velocity', 'Velocity Data'])
plt.xlabel('time')
plt.grid()
plt.show()
