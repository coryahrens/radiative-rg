"""
Linear diffusive wave problem:

u_t - div(grad(u)) = 0,
u(0,t) = 1
u(1,t) = 0
u(x,0) = 1.0e-9

Method: DG in spase and either
backward Euler or Crank-Nicolson
on time.

"""
from dolfin import *
import numpy as np
import sys
import matplotlib.pylab as plt

#set_log_level(WARNING)

# Create mesh and define function space
mesh  = UnitIntervalMesh(100)
nu    = FacetNormal(mesh)
h     = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
V     = FunctionSpace(mesh, 'DG', 1)

T  = 1.0
dt = 0.01


# Define boundary conditions
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x<DOLFIN_EPS
class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return x>1.-DOLFIN_EPS
bc_left = DirichletBC(V, Constant(1.0), LeftBoundary(),'pointwise')
bc_right = DirichletBC(V, Constant(0.0), RightBoundary(), 'pointwise')
bcs = [bc_left, bc_right]

# Define initial condition
class fun(Expression):
    def eval(self, values, x):
        values[0] = 1.0e-4 #


# initial condition
u_old = interpolate(fun(),V)


# Define variational problem
v     = TestFunction(V)
u     = TrialFunction(V)

#Set cr = True --> Crank-Nicolson method; cr = False --> Backward Euler
cr = False
alpha = 4.0
if cr:
  a = dot(grad(u), grad(v)) *dx\
    - dot(jump(u,nu), avg(grad(v))) *dS\
    - dot(avg(grad(u)), jump(v,nu)) *dS\
    + alpha/h_avg * dot(jump(u,nu), jump(v,nu))*dS
  a_old = dot(grad(u_old), grad(v)) *dx\
    - dot(jump(u_old,nu), avg(grad(v))) *dS\
    - dot(avg(grad(u_old)), jump(v,nu)) *dS\
    + alpha/h_avg * dot(jump(u_old,nu), jump(v,nu))*dS
  F  = (u-u_old)*v*dx + 0.5*dt*(a + a_old)
else:
  a = dot(grad(u), grad(v)) *dx\
  - dot(jump(u,nu), avg(grad(v))) *dS\
  - dot(avg(grad(u)), jump(v,nu)) *dS\
  + alpha/h_avg * dot(jump(u,nu), jump(v,nu))*dS
  F  = (u-u_old)*v*dx + dt*a


u_new = Function(V) 
F     = action(F, u_new)
J     = derivative(F, u_new, u)

# Compute solution
problem = NonlinearVariationalProblem(F, u_new, bcs, J)
solver  = NonlinearVariationalSolver(problem)


# Nonlinear solver parameters
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-9
prm['newton_solver']['relative_tolerance'] = 1E-8
prm['newton_solver']['maximum_iterations'] = 30
prm['newton_solver']['relaxation_parameter'] = 1.0
prm['newton_solver']['linear_solver'] = 'gmres'
prm['newton_solver']['preconditioner'] = 'ilu'
prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9
prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['krylov_solver']['maximum_iterations'] = 1000
prm['newton_solver']['krylov_solver']['gmres']['restart'] = 40
prm['newton_solver']['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0

# plotting
plt.ion()
fig = plt.figure(1)	
plt.ioff()

# plot func
def plots(u,xvals,t):
    plt.cla()
    plt.plot(xvals,u)
    plt.ylim(-0.1, 1.1)
    plt.xlim( 0.0, 1.0)
    plt.xlabel(r'$x$',fontsize = 15)
    plt.ylabel(r'$u$',fontsize = 15)
    plt.title(r'Time: $t$ = ' + str(t).ljust(4, str(0)))
    plt.grid("on")
    fig.canvas.draw()
    return

# extract mesh coordinates
xvals = mesh.coordinates()

t = dt
while t <= T:
    solver.solve()
    t += dt
    u_old.assign(u_new)
    plots(u_new.compute_vertex_values(),xvals,t)


plt.show()

