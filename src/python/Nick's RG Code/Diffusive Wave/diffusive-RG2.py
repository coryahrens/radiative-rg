"""
RG iteration for the diffusive wave 
equation with Dirichlet conditions:
    u_t - div(q(u) grad u)= 0
    q(u) = u^5
    u(x,0) = 1.0e-3, 0<x<=500
    u(x,0) = 1     , x = 0
Solver: IIPDG
"""
from dolfin import *
import numpy
import matplotlib.pyplot as plt

# suppress output (but not warnings/errors) 
set_log_level(30)

# this helps with dolfin's strange reordering 
# of function/array values.
parameters["reorder_dofs_serial"] = False

# RG params
L      = 1.05
numIts = 500
alphas = numpy.zeros((numIts,1))
alf    = 0. #initialize zero-th alpha
betas  = numpy.zeros((numIts,1))

# mesh params
nx     = 1000 # num grip points
intMin = 0. # left boundary
intMax = 100  # right boundary   

# update for beta
def getBeta(alpha):
    return (1.-5.*alpha)/2.

def q(u):
    return (1.0e-5 + u**5.)

# Define initial condition
class fun(Expression):
    def eval(self, values, x):
        if numpy.abs(x[0]) == 0.0:
            values[0] = 1.
        else:
            values[0] = 1.0e-3

# apply Initial condition
mesh = IntervalMesh(nx, intMin,intMax)
xvals = mesh.coordinates()
nu    = FacetNormal(mesh)
h     = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
V     = FunctionSpace(mesh, 'DG', 1)
V_CG     = FunctionSpace(mesh, 'CG', 1)
u_old  = interpolate(fun(), V)

# Define boundary conditions
tol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < tol

def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-intMin*L**alf) < tol

Gamma_0 = DirichletBC(V, Constant(1.0), left_boundary,'pointwise')
Gamma_1 = DirichletBC(V, Constant(1.0e-3), right_boundary,'pointwise')
bcs = [Gamma_0, Gamma_1]

# mesh rescaling
class moving_radius(Expression):
    def eval(self, value, x):
        bet = getBeta(alf)
        value[0] = (L**-bet-1)*x[0]

T  = L-1     # simulation time
dt = 0.01    # time step

### Main Loop ###
for kk in range(0,numIts):

    # save the initial condition
    utmp = interpolate(u_old,V_CG)
    uold = utmp.vector().array()

    # Define variational problem
    v     = TestFunction(V)
    u     = TrialFunction(V)

    alpha = 1000.0
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
    prm = solver.parameters

    t = dt
    while t <= T:
        solver.solve()
        t += dt
        u_old.assign(u_new)

    # update/store alpha
    utmp2 = interpolate(u_new,V_CG)
    unew = utmp2.vector().array()
    alf  = numpy.log(numpy.abs(uold).max()/\
           numpy.abs(unew).max())/numpy.log(L)
    alphas[kk] = alf
    bet        = getBeta(alf)
    betas[kk]  = bet

    # rescale amplitude
    utmp3 = interpolate(u_new,V_CG)
    utmp4 = L**alf*utmp3.compute_vertex_values() # extract and rescale values
    utmp3.vector().set_local(utmp4.ravel())      # assign rescaled values to function
    u_old = interpolate(utmp3,V)                 # interpolate to discontinuous func space

    # rescale mesh
    motion = moving_radius()
    mesh.move(motion)

    print kk+1 , alf , bet

# plots
plt.figure(1)
xvals = mesh.coordinates()
utmp5 = interpolate(u_old,V_CG)
u = utmp5.vector().array()
plt.plot(xvals,u,linewidth=3)
plt.ylim(0, 1.01)
plt.xlim(xvals.min(),xvals.max())
plt.xlabel(r'$x$',fontsize = 15)
plt.ylabel(r'$u$',fontsize = 15)
plt.title('Fixed point of the RG map')
plt.grid("on")

plt.figure(2)
plt.plot(alphas,'b-',linewidth=3)
plt.plot(betas,'r-.',linewidth=3)
plt.xlabel(r'$n$',fontsize = 15)
plt.title('Conergence of RG Scaling Factors')
plt.legend([r'$\alpha$',r'$\beta$'])
plt.grid("on")

plt.show()


