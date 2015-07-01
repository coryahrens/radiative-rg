"""
Non-Linear diffusive wave problem:

u_t - div(u^8 grad(u)) = 0,
u(0,t) = 1
u(1,t) = 0
u(x,0) = max(1-2x,0)

Method: IIPDG in space and
backward Euler in time.

"""
from dolfin import *
import numpy as np
import sys
import matplotlib.pylab as plt

#set_log_level(WARNING)

# Create mesh and define function space
mesh  = UnitIntervalMesh(400)
nu    = FacetNormal(mesh)
h     = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
V     = FunctionSpace(mesh, 'DG', 2)
T  = 2.0
dt = 0.01


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

# Define initial condition
class fun(Expression):
    def eval(self, values, x):
        values[0] = 1.0e-3  #max(1.-2.*x[0],0)


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

