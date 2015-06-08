"""
RG iteration for the diffusion 
equation with Dirichlet conditions:
    u_t = u_xx
    u(-0.5) = u(0.5) = 0 
"""
from dolfin import *
import numpy
import matplotlib.pyplot as plt

# RG params
L      = 1.05
numIts = 200
alphas = numpy.zeros((numIts,1))
xscale = 1. # initialize scaling factor

# mesh params
nx     = 1000 # num grip points
intMin = -500 # left boundary
intMax = 500  # right boundary   

# Define initial condition
class fun(Expression):
    def eval(self, values, x):
        if numpy.abs(x[0]) <= 0.5:
            values[0] = cos(pi*x[0])
        else:
            values[0] = 0

# define the Dirichlet boundary
class Boundary(SubDomain): 
    def inside(self, x, on_boundary):
        return on_boundary

# mesh rescaling
class moving_radius(Expression):
    def eval(self, value, x):
        value[0] = (L**-0.5-1)*x[0]

T  = L-1     # simulation time
dt = 0.01    # time step

# Initial condition
mesh = IntervalMesh(nx, intMin,intMax)
V    = FunctionSpace(mesh, 'Lagrange', 1)
u_1  = interpolate(fun(), V)

### Main Loop ###
for kk in range(0,numIts):

    # save the initial condition
    uold = u_1.vector().array()

    # Laplace term
    u   = TrialFunction(V)
    v   = TestFunction(V)
    a_K = inner(nabla_grad(u), nabla_grad(v))*dx

    # "Mass matrix" term
    a_M = u*v*dx

    M = assemble(a_M)
    K = assemble(a_K)
    A = M + dt*K

    # Compute solution
    u = Function(V)
    t = dt
    boundary = Boundary()
    bc = DirichletBC(V, 0, boundary)

    ## time loop ##
    while t <= T:
        b = M*u_1.vector()
        bc.apply(A, b)
        solve(A, u.vector(), b)
    
        t += dt
        u_1.assign(u)

    # update/store alpha
    unew = u_1.vector().array()
    alf  = numpy.log(numpy.abs(uold).max()/\
           numpy.abs(unew).max())/numpy.log(L)
    alphas[kk] = alf

    # rescale amplitude
    utmp = L**alf*u_1.compute_vertex_values() # extract and rescale values
    u_1.vector().set_local(utmp.ravel())      # assign rescaled values to function

    # rescale mesh
    motion = moving_radius()
    mesh.move(motion)

    # function space and IC on rescaled mesh
    #V    = FunctionSpace(mesh, 'Lagrange', 1)
    #u_1 = interpolate(u_1, V)

    print kk+1 , alf

# plots
plt.figure(1)
xvals = mesh.coordinates()
u = u_1.vector().array()
plt.plot(xvals,u,linewidth=3)
plt.ylim(0, 1.01)
plt.xlim(xvals.min(),xvals.max())
plt.xlabel(r'$x$',fontsize = 15)
plt.ylabel(r'$u$',fontsize = 15)
plt.title('Fixed point of the RG map')
plt.grid("on")

plt.figure(2)
plt.plot(alphas,linewidth=3)
plt.xlabel(r'$n$',fontsize = 15)
plt.ylabel(r'$\alpha_n$',fontsize = 15)
plt.title(r'Conergence of $\alpha$')
plt.grid("on")

plt.show()


