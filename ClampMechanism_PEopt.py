# Optimierung eines Mechnismus von Alberto Rossi
#
# Zielfunktion: min(Fdriver)
#
# Optimierungsvariablen: a1, a2, a3, a4, a5, xB, yB0, xE, yE
#
from sympy import symbols, Matrix, cos, sin, solve
from sympy.physics.mechanics import *

Vector.simplify = True
Matrix.simplify = True

t1, t2, t3, u = dynamicsymbols('t1 t2 t3 u')
t1d, t2d, t3d, ud = dynamicsymbols('t1 t2 t3 u', 1)
t1dd, t2dd, t3dd, udd = dynamicsymbols('t1 t2 t3 u', 2)
a1, a2, a3, a4, a5 = symbols('a1 a2 a3 a4 a5')
xB, xE, yB0, yE = symbols('xB xE yB0 yE')
Fa, Fc = symbols('Fa Fc')

N = ReferenceFrame('N')
A = N.orientnew('A', 'Axis', [t1, N.z])
#B = A.orientnew('B', 'Axis', [t2, A.z]) # relativer Winkel
B = N.orientnew('B', 'Axis', [t2, N.z]) # absoluter Winkel


P0= Point('P0')

PA = P0.locatenew('PA', a1 * A.x)

PB = PA.locatenew('PB', a2 * B.x)

P0.set_vel(N, 0)

print(' ')

print('velocity point A')
vprint(PA.vel(N))

print('velocity point B')
vprint(PB.vel(N))

print('coordinates point B')
vprint(PB.pos_from(P0))
vprint(dot(PB.pos_from(P0),N.x))
vprint(dot(PB.pos_from(P0),N.y))

PC = PA.locatenew('PC', (a2+a3)* B.x)

C = N.orientnew('C', 'Axis', [t3, N.z]) # absoluter Winkel

PD = PC.locatenew('PD', a4 * C.x)

print('coordinates point D')
vprint(PD.pos_from(P0))
vprint(dot(PD.pos_from(P0),N.x))
vprint(dot(PD.pos_from(P0),N.y))

Meq = a2*B.x.cross(Fa*N.y) - (a2+a3)*Fc/sin(t3)*B.x.cross(C.x)

print('moment equation')
vprint(Meq)

Fdrive = solve(Meq.dot(N.z), Fa)
print('driving force')
vprint(Fdrive)

# kinematic constraints of the mechanism
eqPBx = dot(PB.pos_from(P0),N.x) - xB # horizontal position of point B fixed
eqPEx = dot(PD.pos_from(P0),N.x) + a5 - xE # horizontal position of point E fixed
eqPBy = dot(PB.pos_from(P0),N.y) - yB0 - u # final vertical position of driver
print('constraint equations')
vprint(eqPBx)
vprint(eqPEx)
vprint(eqPBy)

# clamper position
eqPDy = dot(PD.pos_from(P0),N.y) # vertical position of clamper
print('vertical clamper position')
vprint(eqPDy)


# =================================================================
# numerical evaluation

from sympy import Dummy, lambdify
from numpy import array, hstack, zeros, ones, linspace, nditer, pi
from numpy import cos, sin
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt

#Parameter definitions
parameters = [Fc, u] # constant parameters
optvariables = [a1, a2, a3, a4, a5, xB, yB0, xE, yE] # optimization variables
nopt = len(optvariables) # number of optimization variables
parameters_vals = [50., 10.] # values of constants
optvariables_vals = [] # initial guess

# dynamic symbols (position variables)
dynamic =  [t1, t2, t3]
nvar = len(dynamic) # number of state variables

#Create a list of dynamic symbols for simulation
dummy_symbols = [Dummy() for i in dynamic]  # Create a dummy symbol for each variable
dummy_dict = dict(zip(dynamic, dummy_symbols))

# substitute into nonlinear equations
eqPBx_subs = eqPBx.subs(dummy_dict)
eqPBx_func = lambdify(dummy_symbols + optvariables + parameters, eqPBx_subs)
eqPEx_subs = eqPEx.subs(dummy_dict)
eqPEx_func = lambdify(dummy_symbols + optvariables + parameters, eqPEx_subs)
eqPBy_subs = eqPBy.subs(dummy_dict)
eqPBy_func = lambdify(dummy_symbols + optvariables + parameters, eqPBy_subs)
eqPDy_subs = eqPDy.subs(dummy_dict)
eqPDy_func = lambdify(dummy_symbols + optvariables + parameters, eqPDy_subs)
Fdrive_subs = Fdrive[0].subs(dummy_dict)
Fdrive_func = lambdify(dummy_symbols + optvariables + parameters, Fdrive_subs)

# define function that determines mechanism constraint equations
def mech_con(x, args):
# x: state variables
# args: geometric parameters
#
# returns residui of relevant equations
    arguments = hstack((x, args))     # States, input, and parameters
    eqs = [eqPBx_func(*arguments)]
    eqs.append(eqPEx_func(*arguments))
    eqs.append(eqPBy_func(*arguments))
    return eqs

# define function that determines vertical position at point D (clamper)
def yE(x, args):
# x: state variables
# args: geometric parameters
#
# returns y-position of point D (and E)
    arguments = hstack((x, args))     # States, input, and parameters
    return eqPDy_func(*arguments)

# define function that determines driver force
def F_B_y(x, args):
# x: state variables
# args: geometric parameters
#
# returns vertical force at Point B (=driver force)
    arguments = hstack((x, args))     # States, input, and parameters
    return Fdrive_func(*arguments)

# define goal function
def F_integral(x, args):
# x: optimization variables
# args: geometric parameters
#
# returns numerical integral of vertical force times displacement (=work) at Point B 
# devided by umax*Fa, i.e. average force ratio
    arguments = hstack((x, args))     # States, input, and parameters
    umax = args[-1] # max. travel of point B (from parameters)
    qi = [pi/4., -pi/4., -pi/2.] # initial guess
    uarr = linspace(0., umax, 10) # displacement array
    fi = [] # force values
# loop to determine forces at positions ui
    for ui in nditer(uarr):
        arguments[-1] = ui
        qi = fsolve(mech_con, qi, args=(arguments)) # find positions
        fi.append(F_B_y(qi, arguments)) # driving force

    F_int = 0.
    for f in fi:
        F_int += f

    return F_int*uarr[1]/(umax*args[-2]) # work ratio of steps and total work

def plot_mechanism(xp, args, text=' '):
    arguments = hstack((xp, args))     # States, input, and parameters
    qi = [pi/4., -pi/4., -pi/2.] # initial guess
    q_end = fsolve(mech_con, qi, args=(arguments)) # find end positions
    arguments[-1] = 0.
    q_start = fsolve(mech_con, qi, args=(arguments)) # find start positions
    ltx = xp[5]/3. # relatuve coordinates triangle
    lty = 0.5*ltx
# geometry at start
    x_start = zeros(5)
    y_start = zeros(5)
    x_start[1] = x_start[0] + xp[0]*cos(q_start[0]) # point A
    y_start[1] = y_start[0] + xp[0]*sin(q_start[0])
    x_start[2] = x_start[1] + (xp[1]+xp[2])*cos(q_start[1]) # point C
    y_start[2] = y_start[1] + (xp[1]+xp[2])*sin(q_start[1])
    x_start[3] = x_start[2] + xp[3]*cos(q_start[2]) # point D
    y_start[3] = y_start[2] + xp[3]*sin(q_start[2])
    x_start[4] = x_start[3] + xp[4] # point E
    y_start[4] = y_start[3]
    t_B_s = plt.Polygon(((xp[5]+ltx, xp[6]-lty), (xp[5], xp[6]), \
                         (xp[5]+ltx, xp[6]+lty)), color='b')
    t_E_s = plt.Polygon(((x_start[4]+ltx, y_start[4]-lty), (x_start[4], y_start[4]), \
                         (x_start[4]+ltx, y_start[4]+lty)), color='b')
    
# geometry at end
    x_end = zeros(5)
    y_end = zeros(5)
    x_end[1] = x_end[0] + xp[0]*cos(q_end[0]) # point A
    y_end[1] = y_end[0] + xp[0]*sin(q_end[0])
    x_end[2] = x_end[1] + (xp[1]+xp[2])*cos(q_end[1]) # point C
    y_end[2] = y_end[1] + (xp[1]+xp[2])*sin(q_end[1])
    x_end[3] = x_end[2] + xp[3]*cos(q_end[2]) # point D
    y_end[3] = y_end[2] + xp[3]*sin(q_end[2])
    x_end[4] = x_end[3] + xp[4] # point E
    y_end[4] = y_end[3]
    t_B_e = plt.Polygon(((xp[5]+ltx, xp[6]-lty+args[-1]), (xp[5], xp[6]+args[-1]), \
                         (xp[5]+ltx, xp[6]+lty+args[-1])), color='g')
    t_E_e = plt.Polygon(((x_end[4]+ltx, y_end[4]-lty), (x_end[4], y_end[4]), \
                         (x_end[4]+ltx, y_end[4]+lty)), color='g')

    
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-50, 250), ylim=(-150, 150))
    ax.set_aspect('equal')
    ax.grid()

    ax.plot(x_start, y_start, 'o-', lw=2, color="b")
    ax.plot(x_end, y_end, 'o-', lw=2, color="g")

    ax.add_patch(t_B_s)
    ax.add_patch(t_E_s)
    ax.add_patch(t_B_e)
    ax.add_patch(t_E_e)

    plt.title(text)
    
    fig.show()

    return
    

# tuple with parameters (needed in both objective and constraints)
arguments = (parameters_vals,)

# define bounds
b_a1 = (20,30)
b_a2 = (12,30)
b_a3 = (70,90)
b_a4 = (60,80)
b_a5 = (100,120)
b_xB = (25,45)
b_yB0 = (0, 20)
b_xE = (180,220)
b_yE = (-100,-70)
bnds = (b_a1, b_a2, b_a3, b_a4, b_a5, b_xB, b_yB0, b_xE, b_yE)

# define optimization constraint
# vertical start position point E (or D)
def StartPosition_E(x, args):
    arguments = hstack((x, args))     # States, input, and parameters
    qi = [pi/4., -pi/4., -pi/2.] # initial guess
    arguments[-1] = 0. # u is zero at start position
    qi = fsolve(mech_con, qi, args = (arguments)) # find positions
    arguments_all = hstack((qi, arguments))     # States, input, and parameters
    constraintVal = eqPDy_func(*arguments_all) - x[-1]
    constraintVal /= 100.
    return constraintVal
    
# vertical end position point E (or D)
def EndPosition_E(x, args):
    arguments = hstack((x, args))     # States, input, and parameters
    qi = [pi/4., -pi/4., -pi/2.] # initial guess
    qi = fsolve(mech_con, qi, args = (arguments)) # find positions
    arguments_all = hstack((qi, arguments))     # States, input, and parameters
    constraintVal = eqPDy_func(*arguments_all) - (x[-1]+4.*args[-1])
    constraintVal /= 100.
    return constraintVal
cons1 = ({'type':'eq','fun':StartPosition_E, 'args':arguments})
cons2 = ({'type':'eq','fun':EndPosition_E, 'args':arguments})
cons = [cons1, cons2]

# initial guess
x0 = zeros(nopt)
for i in range(0, nopt, 1):
    bnd = bnds[i]
    x0[i] = bnd[0] + 0.6*(bnd[1]-bnd[0])
print('initial guess')
print(x0)
print(F_integral(x0, *arguments))
print(StartPosition_E(x0, *arguments))
print(EndPosition_E(x0, *arguments))

# plot initial configuration
plot_mechanism(x0, parameters_vals, text='initial configuration')

# perform optimization

sol = minimize(F_integral, x0, args=arguments, method='SLSQP', \
               bounds=bnds, constraints=cons, options={'disp':True})

print(sol)
print(sol.values())
print(sol.x)

# plot final configuration
plot_mechanism(sol.x, parameters_vals, text='final configuration')

# plot force variation
arg_init = hstack((x0, parameters_vals))     # States, input, and parameters
arg_fin = hstack((sol.x, parameters_vals))     # States, input, and parameters
umax = arg_init[-1] # max. travel of point B (from parameters)
q_init = [pi/4., -pi/4., -pi/2.] # initial guess
q_fin = [pi/4., -pi/4., -pi/2.] # initial guess
np = 50 # points in grafics
uarr = linspace(0., umax, np)# displacement array
F_init = zeros(np) # force values initial configuration
F_fin = zeros(np) # force values final configuration
yE_fin = zeros(np) # vertical position final configuration

# loop to determine forces and vertical position point E at positions ui
for i in range(0, np, 1):
    arg_init[-1] = uarr[i]
    arg_fin[-1] = uarr[i]
    q_init = fsolve(mech_con, q_init, args=(arg_init)) # find positions
    F_init[i] = F_B_y(q_init, arg_init)
    q_fin = fsolve(mech_con, q_fin, args=(arg_fin)) # find positions
    F_fin[i] = F_B_y(q_fin, arg_fin)
    arguments_all = hstack((q_fin, arg_fin))
    yE_fin[i] = eqPDy_func(*arguments_all)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()

ax.plot(uarr, F_init, '--', color='b', label='initial')
ax.plot(uarr, F_fin, color='g', label='final')

plt.xlabel("stroke")
plt.ylabel("force")
plt.title("force at point B vs stroke")
plt.legend()
    
fig.show()


# plot force and position point E of final configuration
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid()

color = 'k'
ax1.set_xlabel('stroke')
ax1.set_ylabel('force', color=color)
ax1.plot(uarr, F_fin, color='g', label='Fa')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('position', color=color)  # we already handled the x-label with ax1
ax2.plot(uarr, yE_fin, ':', color='r', label='yE')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('force and vertical position point E')

# Adding legend, which helps us recognize the curve according to it's color
fig.legend(loc = 'lower center', bbox_to_anchor=(0.5, 0.15))

fig.tight_layout()  # otherwise the right y-label is slightly clipped

fig.show()

