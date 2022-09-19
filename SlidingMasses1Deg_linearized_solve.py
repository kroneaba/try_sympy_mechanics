from sympy import symbols, Matrix, sin, cos, trigsimp, pi,sqrt
from sympy.physics.mechanics import *

Vector.simplify = True
Matrix.simplify = True

q = dynamicsymbols('q')
qd = dynamicsymbols('q', 1)
qdd = dynamicsymbols('q', 2)
l, m1, m2, g, c = symbols('l m1 m2 g c', positive=True)
t = dynamicsymbols._t

N = ReferenceFrame('N') # inertial coordinate system

O = Point('O')
O.set_vel(N,0*N.x) # zero velocity (tbd as vector)

P1 = O.locatenew('P1', l * sin(q) * N.x)
P2 = O.locatenew('P2', l * cos(q) * N.z)

# vprint(P1.vel(N))
# vprint(P2.vel(N))

ParM1 = Particle('ParM1', P1, m1)
ParM2 = Particle('ParM1', P2, m2)

ParM1.potential_energy = 1/2 * c * dot(P1.pos_from(O),P1.pos_from(O))
ParM2.potential_energy = m2 * g * dot(P2.pos_from(O),N.z) # result is a scalar

L = Lagrangian(N, ParM1, ParM2)

lm = LagrangesMethod(L, [q], bodies=[ParM1, ParM2])
lm.form_lagranges_equations()

print('equations:')
vprint(lm.form_lagranges_equations())
print('mass matrix:')
vprint(lm.mass_matrix_full)
print('force matrix:')
vprint(lm.forcing_full)

lm_lin = lm.to_linearizer(q_ind=[q], qd_ind=[qd])

# linearization about q=0
op_point = {q: 0, qd: 0}

M, A, B = lm_lin.linearize(op_point=op_point)

print(' ')
print('linear mass matrix:')
vprint(M)
print('linear force matrix:')             
vprint(A)
print('linear matrix B:')  
vprint(B)

# create symbolic solution
import sympy.solvers.ode
om = symbols('om', positive=True)
# force matrix is on rhs, thus its sign must be reversed, if it is put on lhs
#eq = sympy.solvers.ode.dsolve(M[1,1]*qdd-A[1,0]*q, ics={q.subs(t, 0): pi/12, qd.subs(t, 0): 0})
eq = sympy.solvers.ode.dsolve(qdd+om*om*q, ics={q.subs(t, 0): pi/12, qd.subs(t, 0): 0})

eq = trigsimp(eq)

print(' ')
print('solution of DEQ:')
vprint(eq)
vprint(eq.rhs) # only right hand side of equation

# energies
EpotM1 = potential_energy(ParM1)
EkinM1 = kinetic_energy(N, ParM1)
EpotM2 = potential_energy(ParM2)
EkinM2 = kinetic_energy(N, ParM2)

print(' ')
print('potential energy:')
vprint(EpotM1)
vprint(EpotM2)
print('kinetic energy:')
vprint(EkinM1)
vprint(EkinM2)

# =================================================================
# numerical evaluation

from sympy import Dummy, lambdify, diff
from numpy import array, hstack, zeros, ones, linspace, pi, empty, amax, amin
from numpy.linalg import solve
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Parameter definitions
parameters = [l, m1, m2, g, c]
parameter_vals = [0.8, 15., 10., 9.81, 500.0] #Values of constants

# numerical evaluation of omega
om_symb = sqrt(-A[1,0]/M[1,1])
print(' ')
vprint(om_symb)
om_func = lambdify(parameters, om_symb)
om_num = om_func(*parameter_vals)
freq = om_num/(2.*pi)
period = 1./freq
print('om = ',om_num, 'period = ', period)
param_sol = [om]
param_sol_vals = [om_num]

# function of solution of linear deq and its derivatives
sol_time = [t]
dummy_t = [Dummy() for i in sol_time]  # Create a dummy symbol for each variable
dummy_dict_t = dict(zip(sol_time, dummy_t))
sol_q = eq.rhs.subs(dummy_dict_t)
eqd = eq.rhs.diff(t)
eqdd = eq.rhs.diff(t,2)
sol_qd =eqd.subs(dummy_dict_t)
sol_qdd =eqdd.subs(dummy_dict_t)
q_func = lambdify(dummy_t+param_sol, sol_q)
qd_func = lambdify(dummy_t+param_sol, sol_qd)
qdd_func = lambdify(dummy_t+param_sol, sol_qdd)

# dynamic symbols in deq
dynamic =  [q, qd, qdd]

#Create a list of dynamic symbols for simulation
dummy_symbols = [Dummy() for i in dynamic]  # Create a dummy symbol for each variable
dummy_dict = dict(zip(dynamic, dummy_symbols))

# substitute into nonlinear deq which gives residualfunction
res = lm.form_lagranges_equations().subs(dummy_dict)
res_func = lambdify(dummy_symbols + parameters, res)

# define a function that returns the position variable and its derivatives
def solution(t, args):
    """Returns state variables.

    Parameters
    ----------
    t: ndarray with time points
    npts: number of samplimg points
    args: ndarray with the constants.

    Returns
    -------
    u, ud, udd: ndarrays with solution of position variables and its derivatives
    
    """
    npts = t.size
    
    u = empty([npts])
    ud = empty([npts])
    udd = empty([npts])

    for i in range(0,len(t),1):
        arguments = hstack((t[i], args))     # States, input, and parameters
        u[i] = q_func(*arguments)
        ud[i] = qd_func(*arguments)
        udd[i] = qdd_func(*arguments)

    return u, ud, udd

# define a function that returns the residuum of the nonlinear deq
def residuum(t, u, ud, udd, args):
    """Returns the residuum.

    Parameters
    ----------
    t: ndarray tith timepoints
    u, ud, udd : state variable and its derivatives
    args : ndarray with the constants.

    Returns
    -------
    res : ndarray with residuum at time points
    
    """
    npts = t.size
    
    res = empty([npts])

    for i in range(0,len(t),1):
        variables =[u[i], ud[i], udd[i]]
        arguments = hstack((variables, args))     # States, input, and parameters
        res[i] = res_func(*arguments)

    return res

# linear solution and residuals
npts = 201 # number of sample points
t_arr = linspace(0., period/2., npts) # time vector (half period)
y, yd, ydd = solution(t_arr, param_sol_vals)
res = residuum(t_arr, y, yd, ydd, parameter_vals)

print(' ')
print('solution (t, q, res) (every 10th step):')
for i in range(0,npts,10):
    print(t_arr[i],y[i],res[i])
print(' ')
print('max. and min. residuum:')
print('max. res = %10.3E min. res = %10.3E' % (amin(res), amax(res)))
print(' ')

# plot the results
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('pos. angle', color=color)
ax1.plot(t_arr, y, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('residuum', color=color)  # we already handled the x-label with ax1
ax2.plot(t_arr, res, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('position angle & residuum vs time')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('pos. angle')
ax1.set_ylabel('residuum', color=color)
ax1.plot(y, res, color=color)
ax1.tick_params(axis='y', labelcolor=color)

plt.title('residuum vs position angle')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()

# energy 
# substitute into nonlinear deq which gives residualfunction
Epot = EpotM1 + EpotM2
Ekin = EkinM1 + EkinM2
Etot = Epot + Ekin
Etot_subs = Etot.subs(dummy_dict)
Etot_func = lambdify(dummy_symbols + parameters, Etot_subs)

# define a function that returns the total energy of the nonlinear deq
def TotEnergy(t, u, ud, udd, args):
    """Returns the total energy.

    Parameters
    ----------
    t: ndarray tith timepoints
    u, ud, udd : state variable and its derivatives
    args : ndarray with the constants.

    Returns
    -------
    res : ndarray with enrgies at time points
    
    """
    npts = t.size
    
    res = empty([npts])

    for i in range(0,len(t),1):
        variables =[u[i], ud[i], udd[i]]
        arguments = hstack((variables, args))     # States, input, and parameters
        res[i] = Etot_func(*arguments)

    return res

Etot_arr = TotEnergy(t_arr, y, yd, ydd, parameter_vals)

Etot_ref = Etot_arr[0]

Etot_rel = empty([npts])

for i in range(0,len(t_arr),1):
     Etot_rel[i] = (Etot_arr[i]-Etot_ref) / Etot_ref # relative energy with respect to initial energy 

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('pos. angle')
ax1.set_ylabel('relative energy', color=color)
ax1.plot(y, Etot_rel, color=color)
ax1.tick_params(axis='y', labelcolor=color)

plt.title('relative total energy vs position angle')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()
