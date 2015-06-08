"""
Diffusion equation with Dirichlet conditions:
    u_t = a*u_xx
    u(-0.5) = u(0.5) = 0 
"""

from dolfin import *
import numpy
import matplotlib.pyplot as plt
import time

# diffusion coeff
a = 0.25

# Create mesh and define function space
nx   = 200
xscale = 1.
mesh = IntervalMesh(nx, -0.5*xscale,0.5*xscale)
V    = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
u0 = Expression('cos(3.14159265*x[0])')

class Boundary(SubDomain):  # define the Dirichlet boundary
    def inside(self, x, on_boundary):
        return on_boundary

boundary = Boundary()
bc = DirichletBC(V, 0, boundary)

# Initial condition
u_1 = interpolate(u0, V)

T  = 1       # total simulation time
dt = 0.01    # time step

# Define variational problem

# Laplace term
u   = TrialFunction(V)
v   = TestFunction(V)
a_K = a*inner(nabla_grad(u), nabla_grad(v))*dx

# "Mass matrix" term
a_M = u*v*dx

M = assemble(a_M)
K = assemble(a_K)
A = M + dt*K

# extract mesh coordinates
xvals = mesh.coordinates()

# plotting
plt.ion()
fig = plt.figure(1)	
plt.ioff()
 
# plot func
def plots(u,xvals,t):
    u = u.vector().array()
    plt.cla()
    plt.plot(xvals,u,linewidth=3)
    plt.ylim(0, 1.01)
    plt.xlim(-0.5,0.5)
    plt.xlabel(r'$x$',fontsize = 15)
    plt.ylabel(r'$u$',fontsize = 15)
    plt.title(r'FEM Solution to the 1D Heat Equation: $t$ = '\
    + str(t).ljust(4, str(0)))
    plt.grid("on")
    fig.canvas.draw()
    #time.sleep(.1)
    return

#plot IC
plots(u_1,xvals,0)

# Compute solution
u = Function(V)
t = dt
while t <= T:
    b = M*u_1.vector()
    u0.t = t
    bc.apply(A, b)
    solve(A, u.vector(), b)
    
    t += dt
    u_1.assign(u)
    plots(u_1,xvals,t)

plt.show()
