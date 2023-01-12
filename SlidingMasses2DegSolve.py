from sympy import symbols, Matrix
from sympy.physics.mechanics import *

Vector.simplify = True
Matrix.simplify = True

q1, q2 = dynamicsymbols('q1 q2')
q1d, q2d = dynamicsymbols('q1 q2', 1)
q1dd, q2dd = dynamicsymbols('q1 q2', 2)
l, m1, m2, g, c = symbols('l m1 m2 g c')

N = ReferenceFrame('N')

O = Point('O')
P1 = O.locatenew('P1', q1 * N.x)
P2 = O.locatenew('P2', q2 * N.z)

O.set_vel(N, 0)
P1.set_vel(N, q1d * N.x)
P2.set_vel(N, q2d * N.z)

ParM1 = Particle('ParM1', P1, m1)
ParM2 = Particle('ParM1', P2, m2)

ParM1.potential_energy = 0.5 * c * q1*q1 # zero force at q1=0
ParM2.potential_energy = m2 * g * q2

L = Lagrangian(N, ParM1, ParM2)

Phi = Matrix([q1*q1+q2*q2-l*l])
#Phi = Matrix([l*l-q1*q1-q2*q2])

lm = LagrangesMethod(L, [q1, q2], bodies=[ParM1, ParM2], hol_coneqs=Phi)
lm.form_lagranges_equations()

print('equations:')
vprint(lm.form_lagranges_equations())
print('mass matrix:')
vprint(lm.mass_matrix_full)
print('forcing:')
vprint(lm.forcing_full)

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

from sympy import Dummy, lambdify
from numpy import array, hstack, zeros, ones, linspace, pi
from numpy import matmul, transpose, append, empty
from numpy.linalg import solve
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, pi, atan2

#Parameter definitions
parameters = [l, m1, m2, g, c]
parameter_vals = [0.8, 15., 10., 9.81, 500.0] #Values of constants

# dynamic symbols in deq
dynamic =  [q1, q2, q1d, q2d]
nvar = len(dynamic) # number of state variables
ndof = int(nvar/2) # number of position coordinates
ncon = Phi.shape[0] # number of constraints

#Create a list of dynamic symbols for simulation
dummy_symbols = [Dummy() for i in dynamic]  # Create a dummy symbol for each variable
dummy_dict = dict(zip(dynamic, dummy_symbols))

# substitute into nonlinear deq which gives residualfunction
M_subs = lm.mass_matrix_full.subs(dummy_dict)
M_func = lambdify(dummy_symbols + parameters, M_subs)
h_subs = lm.forcing_full.subs(dummy_dict)
h_func = lambdify(dummy_symbols + parameters, h_subs)
# constraint equations
Phi_subs = Phi.subs(dummy_dict)
Phi_func = lambdify(dummy_symbols + parameters, Phi_subs)
Dsub = lm.mass_matrix_full[nvar:nvar+ncon,ndof:nvar] # Jacobian of constraint
D_subs = Dsub.subs(dummy_dict)
D_func = lambdify(dummy_symbols + parameters, D_subs)

# define a function that returns the derivatives of the states given the current state and time
def eq_mot(x, t, alpha, beta, args):
    """Returns the derivatives of the states.

    Parameters
    ----------
    x : ndarray, shape(2 * (n + 1))
        The current state vector.
    t : float
        The current time.
    alpha, beta: float
        
    args : ndarray
        The constants.

    Returns
    -------
    dx : ndarray, shape(2 * (n + 1))
        The derivative of the state.
    
    """
    arguments = hstack((x[:nvar], args))     # States, input, and parameters
    hi = h_func(*arguments)
    # adding penalty terms in case of constraint violation
    penalty = 2*alpha*matmul(D_func(*arguments),x[ndof:nvar]) +\
            beta*beta*Phi_func(*arguments)[0:ncon]
    hi[nvar:nvar+ncon] -=  penalty[0:ncon]
    dx = array(solve(M_func(*arguments), hi)).T[0] # Solving for the derivatives

    return dx

# time array for solution
npts = 201 # number of sample points
# determine period 
# om = sqrt((1.0*c*l**2 - g*l*m2)/(l**2*m1)) from 1 dof analysis
om = sqrt((1.0*parameter_vals[4]*parameter_vals[0]*parameter_vals[0] - \
            parameter_vals[3]*parameter_vals[0]*parameter_vals[2])/ \
            (parameter_vals[0]*parameter_vals[0]*parameter_vals[1]))
period = 2.0*pi / om
t_arr = linspace(0., period, npts) # time array

# initial conditions
phi0 = pi / 12.
lrod = parameter_vals[0]
x0 = hstack((lrod*sin(phi0)*ones(1),lrod*cos(phi0)*ones(1), zeros(2) )) # initial conditions, q and qd
# determination of initial lambda
# note: though initial condition satisfy constraint equation, lambda is not necessaril#y zero.
# Since lambda is ~ force, lambda has some amount to due weight force (counterbalanced by the spring)
# procedure acc. to Nikravesh equ. 13.23
arguments = hstack((x0, parameter_vals))     # States, input, and parameters

#matrix inversion failed with numpy. Therefore, sub-mass matrix is determined symbollically first.
Mred = lm.mass_matrix_full[ndof:nvar,ndof:nvar]
Mredinv = Mred.inv()
Minv_subs = Mredinv.subs(dummy_dict)
Minv_func = lambdify(dummy_symbols + parameters, Minv_subs)

M0inv = Minv_func(*arguments)# mass submatrix
hsub = h_func(*arguments)[ndof:nvar] # force subvector
gamma = h_func(*arguments)[nvar] # rhs acceleration constraint
D0 = D_func(*arguments)# Jacobian of constraints submatrix

DMinv = matmul(D0,M0inv)

lam_rhs = gamma - matmul(DMinv,hsub)
DMinvDT = matmul(DMinv,transpose(D0))

lam = array(solve(DMinvDT, lam_rhs)).T[0] # solving for lambda

x0 = append(x0, lam) # full vector with initial coordinates

# nonlinear solution
alpha=5.;beta=5. # parameters constraint violation penalty
ysys = odeint(eq_mot, x0, t_arr, args=(alpha, beta, parameter_vals))    # Actual integration

print(' ')
print('nonlinear solution (t, q, qd) (every 10th step):')
for i in range(0,npts,10):
    print(t_arr[i],ysys[i,0],ysys[i,1],ysys[i,2],ysys[i,3],ysys[i,4])
print(' ')

# determination of quantities fpr assessment
phi = empty([npts])
Phi_err = empty([npts])
Frod = empty([npts])
Frod_x = empty([npts])
Frod_y = empty([npts])
Fc = empty([npts])
ydsys = empty([npts,nvar+ncon])

for i in range(0,npts,1):
    phi[i] = atan2(ysys[i,0],ysys[i,1]) # angle between vertical axis and rod
    arguments = hstack((ysys[i,:nvar], parameter_vals))     # States, input, and parameters
    Phi_err[i] = Phi_func(*arguments)/(lrod*lrod)
    ydsys[i,:] = eq_mot(ysys[i,:], t_arr[i], 0., 0., parameter_vals) # recover accel. & lambda
    Frod[i] = 2.*ydsys[i,nvar] * lrod # normal load of rod = 2*lambda*l
    Frod_x[i] = 2.*ydsys[i,nvar] * ysys[i,0] # component x of normal load of rod =lambda*q1
    Frod_y[i] = 2.*ydsys[i,nvar] * ysys[i,1] # component y of normal load of rod =lambda*q1
    Fc[i] = parameter_vals[4] * ysys[i,0] # spring force
    
# plot the results - position variable
# Plotting both the positions and rod angle simultaneously
fig, ax1 = plt.subplots()

color = 'k'
ax1.set_xlabel('time')
ax1.set_ylabel('position', color=color)
ax1.plot(t_arr, ysys[:, 0], color='g', label='q1')
ax1.plot(t_arr, ysys[:, 1], '--', color='b', label='q2')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('angle', color=color)  # we already handled the x-label with ax1
ax2.plot(t_arr, phi, ':', color='r', label='phi')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('position & rod angle vs time')

# Adding legend, which helps us recognize the curve according to it's color
fig.legend(loc = 'lower right', bbox_to_anchor=(0.85, 0.15))

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()

# plot the results - velocities
plt.plot(t_arr, ysys[:, 2], color='g', label='q1d')
plt.plot(t_arr, ysys[:, 3], '--', color='b', label='q2d')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("time")
plt.ylabel("velocity")
plt.title("velocities q1d and q2d - 1 period")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window 
plt.show()

# plot the results - lambda
plt.plot(t_arr, ydsys[:,nvar], color='b', label='lambda')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("time")
plt.ylabel("lambda")
plt.title("lambda - 1 period")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()

# plot the results - constraint violation error
plt.plot(t_arr, Phi_err, color='b', label='Phi/lrod^2')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("time")
plt.ylabel("relative error")
plt.title("constraint violation error - 1 period")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()

# plot the results - forces
plt.plot(ysys[:,0], Frod, color='k', label='rod')
plt.plot(ysys[:,0], Frod_x, '--', color='g', label='rod x')
plt.plot(ysys[:,0], Frod_y, '-.', color='b', label='rod y')
plt.plot(ysys[:,0], Fc, ':', color='r', label='spring')
plt.plot(ysys[:,0], ydsys[:,nvar], ':', color='m', label='lambda')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("position")
plt.ylabel("force")
plt.title("rod and spring forces vs q1 - 1 period")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()

# energy 
# substitute into nonlinear deq which gives residualfunction
Epot = EpotM1 + EpotM2
Ekin = EkinM1 + EkinM2
Etot = Epot + Ekin
Etot_subs = Etot.subs(dummy_dict)
Etot_func = lambdify(dummy_symbols + parameters, Etot_subs)

# define a function that returns the total energy
def TotEnergy(t, y, nvar, args):
    """Returns the total energy.

    Parameters
    ----------
    t: ndarray tith timepoints
    u, ud, udd : state variable and its derivatives
    args : ndarray with the constants.

    Returns
    -------
    res : ndarray with energies at time points
    
    """
    res = empty([npts])

    for i in range(0,npts,1):
        arguments = hstack((y[i,:nvar], args))      # States, input, and parameters
        res[i] = Etot_func(*arguments)

    return res

Etot_nl = TotEnergy(t_arr, ysys, nvar, parameter_vals)

Etot_ref = Etot_nl[0]

Erel_nl = empty([npts])

for i in range(0,npts,1):
     Erel_nl[i] = (Etot_nl[i]-Etot_ref) / Etot_ref # relative energy with respect to initial energy 


# plot the results - energy balance
plt.plot(t_arr, Erel_nl, color='g', label='total')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("time")
plt.ylabel("relative energy change")
plt.title("energy balance - 1 period")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()
