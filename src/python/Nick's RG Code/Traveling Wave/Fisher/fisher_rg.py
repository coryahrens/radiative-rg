"""
Using RG map to find traveling wave solutions to
Fisher's equation with Dirichlet conditions:
    u_t = u_xx + u*(1-u)
    u(0) = 1, u(50) = 0 
"""
from dolfin import *
import numpy
import matplotlib.pyplot as plt

set_log_level(30) # suppress output (other than warnings) 

# Create mesh and define function space
x_r  = 50
nx   = 500
mesh = IntervalMesh(nx, 0,x_r)
V    = FunctionSpace(mesh, 'Lagrange',1)


# Define initial condition
class fun(Expression):
    def eval(self, values, x):
        if numpy.abs(x[0]) <= 1E-14:
            values[0] = 1
        else:
            values[0] = 0

# Define Dirichlet conditions for x=0 boundary

u_L = Constant(1)

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0]) < tol

Gamma_0 = DirichletBC(V, u_L, LeftBoundary())

# Define Dirichlet conditions for x=x_r boundary

u_R = Constant(0)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0] - x_r) < tol
 
Gamma_1 = DirichletBC(V, u_R, RightBoundary())

bcs = [Gamma_0, Gamma_1]

# Initial condition
u_1 = interpolate(fun(), V)

T  = 10       # total simulation time
dt = 0.01    # time step

# Define variational problem

# Laplace term
u   = TrialFunction(V)
v   = TestFunction(V)
a = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*dx
L = (u_1 + dt*u_1*(1-u_1))*v*dx

A = assemble(a)

# Compute solution to generate traveling wave profile
u = Function(V)
t = dt
while t <= T:
    solve(a==L, u, bcs)
    
    t += dt
    u_1.assign(u)
#plot(u, title = 'Initial Traveling Wave Profile')

u_init = u #save ic
####################################################
# RG Section
####################################################

b  = 0.25    # rg simulation time
dt = 0.01    # time step

# new initial condition
utmp = u.vector().array()  
u_1.vector().set_local(utmp.ravel())
u_1 = interpolate(u_1, V)

# Define variational problem

# Laplace term
u   = TrialFunction(V)
v   = TestFunction(V)
a = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*dx
L = (u_1 + dt*u_1*(1-u_1))*v*dx

A = assemble(a)

# Compute solution
## RG Loop ##
for kk in range(0,3):
    u = Function(V)
    t = dt
    u_prev = u_1.compute_vertex_values() # store previous for comparison
    while t <= b:
        solve(a==L, u, bcs)
    
        t += dt
        u_1.assign(u)

    # know v = 2
    # have b = 0.25 and dx = 0.1
    # thus wave travels d = v*b = 1/2 = 5*dx
    # hence, we shift profile to the left by 5

    utmp = u.compute_vertex_values()
    utmp = numpy.append(utmp[4:-1],numpy.zeros((1,5)))
    utmp = utmp[::-1] # why does it need to be flipped?
    u_1.vector().set_local(utmp.ravel())
    u_1 = interpolate(u_1, V)

###### output ######

u_init_vals = u_init.compute_vertex_values()
u_1_vals = u_1.compute_vertex_values()

xvals = mesh.coordinates()

plt.plot(xvals,u_init_vals,xvals,u_1_vals,xvals,u_prev,'-.')
plt.legend(['Initial Condition','RG Result','Pervious Iterate'])
plt.show()
