"""
Fisher's equation with Dirichlet conditions:
    u_t = D*u_xx + r*u*(1-u)
    u(-10) = u(10) = 0 
"""
from dolfin import *
import numpy
import matplotlib.pyplot as plt
import time

# diffusion coeff
D = 0.125
# growth rate
r = 4.0

# Create mesh and define function space
nx   = 250
xscale = 1.
mesh = IntervalMesh(nx, -10,10)
V    = FunctionSpace(mesh, 'Lagrange',1)


# Define initial condition
class fun(Expression):
    def eval(self, values, x):
        if numpy.abs(x[0]) <= 0.5:
            values[0] = 0.5*cos(pi*x[0])
        else:
            values[0] = 0

# Define boundary conditions
class Boundary(SubDomain):  # define the Dirichlet boundary
    def inside(self, x, on_boundary):
        return on_boundary

boundary = Boundary()
bc = DirichletBC(V, 0, boundary)

# Initial condition
u_1 = interpolate(fun(), V)

T  = 2       # total simulation time
dt = 0.01    # time step

# Define variational problem

# Laplace term
u   = TrialFunction(V)
v   = TestFunction(V)
a = u*v*dx + D*dt*inner(nabla_grad(u), nabla_grad(v))*dx
L = (u_1 + r*dt*u_1*(1-u_1))*v*dx

A = assemble(a)

# extract mesh coordinates
xvals = mesh.coordinates()

# plotting
plt.ion()
fig = plt.figure(1)	
plt.ioff()
 
# plot func
def plots(u,xvals,t):
    u = u.compute_vertex_values()
    plt.cla()
    plt.plot(xvals,u,linewidth=3)
    plt.ylim(0, 1.01)
    #plt.xlim(-0.5,0.5)
    plt.xlabel(r'$x$',fontsize = 15)
    plt.ylabel(r'$u$',fontsize = 15)
    plt.title(r"FEM Solution to Fisher's Equation: $t$ = "\
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
    b = assemble(L)
    bc.apply(A, b)
    solve(A, u.vector(), b)
    
    t += dt
    u_1.assign(u)
    plots(u_1,xvals,t)

plt.show()
