"""
Using RG map to find traveling wave solutions to
Fisher's equation with Dirichlet conditions:
    u_t = u_xx + u*(1-u)(1+nu*u)
    u(0) = 1, u(25) = 0 
where nu in [-1,inf).
"""
from dolfin import *
import numpy
import matplotlib.pyplot as plt

# suppress output (but not warnings/errors) 
set_log_level(30)

# this helps with dolfin's strange reordering 
# of function/array values.
parameters["reorder_dofs_serial"] = False

# parameter for fisher's eqn
nu = 2.0

# Create mesh and define function space
x_r  = 100
nx   = 5000
mesh = IntervalMesh(nx, 0,x_r)
V    = FunctionSpace(mesh, 'Lagrange',1)

# Define initial condition
class fun(Expression):
    def eval(self, values, x):
        if numpy.abs(x[0] - 0.5*x_r) <= 0.01*x_r:
            values[0] = 1.0
        else:
            values[0] = 0.0

# Define Dirichlet conditions for x=0 boundary
u_L = Constant(0.0)
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0]) < tol

Gamma_0 = DirichletBC(V, u_L, LeftBoundary())

# Define Dirichlet conditions for x=x_r boundary
u_R = Constant(0.0)
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0] - x_r) < tol
 
Gamma_1 = DirichletBC(V, u_R, RightBoundary())

bcs = [Gamma_0, Gamma_1]

####################################################
# Initial Solve (for aesthetics mostly)
####################################################

T  = 5.0    # total simulation time
dt = 0.01   # time step

# Initial condition
u_1 = interpolate(fun(), V)

# Define variational problem

u   = TrialFunction(V)
v   = TestFunction(V)
a = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*dx
L = (u_1 + dt*u_1*(1.-u_1)*(1.+nu*u_1))*v*dx

# Compute solution to generate traveling wave profile
u = Function(V)
t = dt
while t <= T:
    solve(a==L, u, bcs)
    
    t += dt
    u_1.assign(u)

u_init = u #save ic

####################################################
# RG Section
####################################################

b   = 0.20    # rg simulation time
dt  = 0.02    # time step

velMax = 18.0   # search interval
velMin = 1.0
vel = (velMax+velMin)/2. # initial guess for velocity

f_a = -0.00135292847249 # delU for vel = 1  (nu = 2)
f_b = 0.056959928594    # delU for vel = 18 (nu = 2)

#f_a = -0.012518629212   # delU for vel = 1  (nu = 0)
#f_b = 0.428039780479    # delU for vel = 18 (nu = 0)

## RG Loop ##
for kk in xrange(0,40):
    
    # Create mesh and define function space
    # given the current guess for vel
    l        = b*vel                   # distance wave front moves
    numUnits = 20.                     # num units we will shift 
    delX     = l/numUnits              # new delta x   
    nx       = int(x_r/delX)           # num x steps
    mesh2    = IntervalMesh(nx, 0,x_r)
    V1       = FunctionSpace(mesh2, 'Lagrange',1)

    # bc's on new mesh
    Gamma_0 = DirichletBC(V1, u_L, LeftBoundary())
    Gamma_1 = DirichletBC(V1, u_R, RightBoundary())
    bcs = [Gamma_0, Gamma_1]

    # initial condition on new mesh
    u_1 = interpolate(u_1,V1)

    # Define variational problem
    u  = TrialFunction(V1)
    v  = TestFunction(V1)
    a1 = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*dx
    L1 = (u_1 + dt*u_1*(1.-u_1)*(1.+nu*u_1))*v*dx

    u = Function(V1)
    t = dt

    # store previous for comparison
    u_prev = u_1.compute_vertex_values()

    # interpolate previous solution to reference grid
    u_n = interpolate(u_1,V)
    un_vals = u_n.compute_vertex_values()

    ## time loop ##
    while t <= b:
        solve(a1==L1, u, bcs)    
        t += dt
        u_1.assign(u)
    
    # shift and assign values to IC
    utmp = u_1.compute_vertex_values()
    utmp = numpy.append(utmp[(numUnits-1):-1],numpy.zeros((1,numUnits)))
    u_1.vector().set_local(utmp.ravel())

    # interpolate solution to reference grid
    u_np1 = interpolate(u_1,V)
    unp1_vals = u_np1.compute_vertex_values()

    # compare solutions at x = 55.
    f_c = un_vals[2750] - unp1_vals[2750]
    
    if (velMax-velMin)/2. < 1E-6:
        print 'Search interval within tolerance.'
        print 'Velocity:', vel
        break
    elif numpy.sign(f_c) == numpy.sign(f_a):
        f_a = f_c
        velMin = vel
        vel = (velMax+velMin)/2.
    else:
        f_b = f_c
        velMax = vel
        vel = (velMax+velMin)/2.

    print kk + 1, vel

###### output ######
u_ics      = interpolate(fun(), V)
u_ics_vals = u_ics.compute_vertex_values()

u_init_vals = u_init.compute_vertex_values()
u_1_vals = u_1.compute_vertex_values()

xvals = mesh2.coordinates()

plt.plot(numpy.linspace(0,x_r,len(u_init_vals)),u_ics_vals, xvals,u_1_vals, numpy.linspace(0,x_r,len(u_prev)),u_prev, '-.')
plt.legend(['Initial Condition','RG Result','Pervious Iterate'])
plt.show()
