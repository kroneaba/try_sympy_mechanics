from sympy import symbols, Matrix, sin, cos, trigsimp, pi
from sympy.physics.mechanics import *

Vector.simplify = True
Matrix.simplify = True

q = dynamicsymbols('q')
qd = dynamicsymbols('q', 1)
qdd = dynamicsymbols('q2', 2)
l, m1, m2, g, c = symbols('l m1 m2 g c')
# t = dynamicsymbols._t

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

# linearization about q=q0
q0 = symbols('q0')
op_point = {q: q0, qd: 0}

M15, A15, B15 = lm_lin.linearize(op_point=op_point)

print(' ')
print('linearization about q=q0')
print('linear mass matrix:')
vprint(M15)
print('linear force matrix:')             
vprint(A15)
print('linear matrix B:')  
vprint(B15)

