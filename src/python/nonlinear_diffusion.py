"""
Nonlinear diffusion equation with Dirichlet conditions:

u_t -div(q(u)*grad(u^2)) = 0,
u = 0 at x=0,1 and
q(u) = (1+u)^2

"""
from dolfin import *
import numpy, sys

set_log_level(WARNING)

# Create mesh and define function space
mesh = UnitIntervalMesh(100)
V    = FunctionSpace(mesh, 'Lagrange',1)

T = 1.0
dt = 0.01


# Define boundary conditions
tol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < tol

def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < tol

Gamma_0 = DirichletBC(V, Constant(0.0), left_boundary)
Gamma_1 = DirichletBC(V, Constant(0.0), right_boundary)
bcs = [Gamma_0, Gamma_1]

# Define initial condition
class fun(Expression):
    def eval(self, values, x):
        values[0] = 1-numpy.abs((2*x[0]-1.))

# nonlinear coefficient
def q(u):
    return (1+u)**2

# initial condition
u_old = interpolate(fun(),V)

plot(u_old,title = "Initial Condition")

# Define variational problem
v     = TestFunction(V)
u     = TrialFunction(V)
F     = (u-u_old)*v*dx + dt*dot(q(u)*grad(u**2), grad(v))*dx
u_new = Function(V) 
F     = action(F, u_new)
J     = derivative(F, u_new, u)

# Compute solution
problem = NonlinearVariationalProblem(F, u_new, bcs, J)
solver  = NonlinearVariationalSolver(problem)

t = dt
while t <= T:
    solver.solve()
    t += dt
    u_old.assign(u_new)

plot(u_new,title = "Solution at t=1")
interactive()
