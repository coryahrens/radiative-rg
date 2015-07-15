"""
Traveling wave RG applied to the
Non-Linear diffusive wave problem:

u_t - div(u^8 grad(u)) = 0,
u(0,t) = 1
u(1,t) = 0
u(x,0) = max(1-2x,0)

Method: IIPDG in space and
backward Euler in time.
"""

from dolfin import *
from ShockWaveSpeed import *
import numpy
import matplotlib.pyplot as plt

# suppress output (but not warnings/errors) 
set_log_level(30)

# this helps with dolfin's strange reordering 
# of function/array values.
parameters["reorder_dofs_serial"] = False

# Create mesh and define function space
x_l   = 0.0
x_r   = 2
mesh  = IntervalMesh(500,x_l,x_r)
xvals = mesh.coordinates()
nu    = FacetNormal(mesh)
h     = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
V     = FunctionSpace(mesh, 'DG', 1)
V_CG     = FunctionSpace(mesh, 'CG', 1)

# Define initial condition
class fun(Expression):
    def eval(self, values, x):
        values[0] = 1.0e-3  #max(1.-2.*x[0],0)

# Define boundary conditions
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x<DOLFIN_EPS
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x>1.-DOLFIN_EPS

bc_left = DirichletBC(V, Constant(1.0), LeftBoundary(),'pointwise')
bc_right = DirichletBC(V, Constant(1.0e-3), RightBoundary(), 'pointwise')
bcs = [bc_left, bc_right]

####################################################
# Initial Solve (for aesthetics mostly)
####################################################

T  = 1.0    # total simulation time
dt = 0.01   # time step

# initial condition
u_old = interpolate(fun(),V)

# Define variational problem
v     = TestFunction(V)
u     = TrialFunction(V)

alpha = 1000.0

def q(u):
  return (1.0e-5 + u**5.)

a = dot(q(u)*grad(u), grad(v))*dx \
  - avg(dot(q(u)*grad(u),nu))*jump(v)*dS\
  + alpha/h_avg*jump(u)*jump(v)*dS

F  = (u-u_old)*v*dx + dt*a

u_new = Function(V) 
F     = action(F, u_new)
J     = derivative(F, u_new, u)

# Compute solution
problem = NonlinearVariationalProblem(F, u_new, bcs, J)
solver  = NonlinearVariationalSolver(problem)


# Nonlinear solver parameters

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-4
prm['newton_solver']['relative_tolerance'] = 1E-5
prm['newton_solver']['maximum_iterations'] = 1000
prm['newton_solver']['relaxation_parameter'] = 0.05 # 0<r<1.0. If Newton doesn't converge set to smaller number 
prm['newton_solver']['linear_solver'] = 'gmres'
prm['newton_solver']['preconditioner'] = 'ilu'
prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-3
prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-4
prm['newton_solver']['krylov_solver']['maximum_iterations'] = 500
prm['newton_solver']['krylov_solver']['gmres']['restart'] = 40
prm['newton_solver']['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0

# Compute solution to generate traveling wave profile
t = dt
while t <= T:
    solver.solve()
    t += dt
    u_old.assign(u_new)


####################################################
# RG Section
####################################################

b   = 0.2    # rg simulation time
dt  = 0.02    # time step

velMax = 3.   # search interval
velMin = 1.
vel = (velMax+velMin)/2. # initial guess for velocity

f_a = -0.084  # delU for vel = 1 
f_b = 0.312   # delU for vel = 3

prm['newton_solver']['maximum_iterations'] = 10000
prm['newton_solver']['relaxation_parameter'] = 0.005 # 0<r<1.0. If Newton doesn't converge set to smaller number 

## RG Loop ##
for kk in xrange(0,25):
    
    # Create mesh and define function space
    # given the current guess for vel
    l        = b*vel                   # distance wave front moves
    numUnits = 75.                     # num units we will shift 
    delX     = l/numUnits              # new delta x   
    nx       = int(abs(x_r-x_l)/delX)           # num x steps
    mesh2    = IntervalMesh(nx,x_l,x_r)
    nu    = FacetNormal(mesh2)
    h     = CellSize(mesh2)
    h_avg = (h('+') + h('-'))/2.0
    V1    = FunctionSpace(mesh2, 'DG',1)
    V1_CG = FunctionSpace(mesh2, 'CG',1)

    # bc's on new mesh
    bc_left = DirichletBC(V1, Constant(1.0), LeftBoundary(),'pointwise')
    bc_right = DirichletBC(V1, Constant(1.0e-3), RightBoundary(), 'pointwise')
    bcs = [bc_left, bc_right]

    # initial condition on new mesh
    u_old = interpolate(u_old,V1)

    # Define variational problem
    v     = TestFunction(V1)
    u     = TrialFunction(V1)

    # Define variational problem
    a = dot(q(u)*grad(u), grad(v))*dx \
      - avg(dot(q(u)*grad(u),nu))*jump(v)*dS\
      + alpha/h_avg*jump(u)*jump(v)*dS

    F  = (u-u_old)*v*dx + dt*a

    u_new = Function(V1) 
    F     = action(F, u_new)
    J     = derivative(F, u_new, u)

    # Compute solution
    problem = NonlinearVariationalProblem(F, u_new, bcs, J)
    solver  = NonlinearVariationalSolver(problem)

    # interpolate previous solution to reference grid
    u_n = interpolate(u_old,V_CG)
    un_vals = u_n.compute_vertex_values()

    ## time loop ##
    t = dt
    while t <= T:
        solver.solve()
        t += dt
        u_old.assign(u_new)
    
    # shift and assign values to IC
    utmp = interpolate(u_old, V1_CG)
    utmp = utmp.compute_vertex_values()
    utmp2 = utmp
    utmp = numpy.append(utmp[(numUnits-1):-1],1.0e-3*numpy.ones((1,numUnits)))
    u_old = Function(V1_CG) 
    u_old.vector().set_local(utmp.ravel())
    u_old = interpolate(u_old, V1)

    # interpolate solution to reference grid
    u_np1 = interpolate(u_old,V_CG)
    unp1_vals = u_np1.compute_vertex_values()

    # get alpha
    t0 = 1.
    t1 = 1. + b
    coeff = getAlpha(t0, un_vals, t1, utmp2, xvals)
    z0,z1 = getShockFrontLocation(un_vals, unp1_vals, xvals)

    # compare solutions at x = 0.5
    f_c = z0-z1
    
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
    print "alpha:", coeff

###### output ######
u_ics      = interpolate(fun(), V)
u_ics_vals = u_ics.compute_vertex_values()

plt.plot(numpy.linspace(x_l,x_r,len(u_ics_vals)),u_ics_vals, numpy.linspace(x_l,x_r,len(unp1_vals)),unp1_vals, numpy.linspace(x_l,x_r,len(un_vals)),un_vals, '-.')
plt.legend(['Initial Condition','RG Result','Pervious Iterate'])
plt.show()
