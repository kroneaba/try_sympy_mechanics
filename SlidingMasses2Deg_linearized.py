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
P2 = O.locatenew('P2', q2 * N.y)

O.set_vel(N, 0)
P1.set_vel(N, q1d * N.x)
P2.set_vel(N, q2d * N.y)

ParM1 = Particle('ParM1', P1, m1)
ParM2 = Particle('ParM1', P2, m2)

ParM1.potential_energy = 0.5 * c * q1*q1
ParM2.potential_energy = m2 * g * q2

L = Lagrangian(N, ParM1, ParM2)

hol_coneqs = [q1*q1+q2*q2-l*l]

lm = LagrangesMethod(L, [q1, q2], bodies=[ParM1, ParM2], hol_coneqs=hol_coneqs)
lm.form_lagranges_equations()

print('equations:')
vprint(lm.form_lagranges_equations())
print('full mass matrix:')
vprint(lm.mass_matrix_full)
print('full force matrix:')
vprint(lm.forcing_full)
print('mass matrix:')
vprint(lm.mass_matrix)
print('force matrix:')
vprint(lm.forcing)

# linearization with q1 as independent variable
lm_lin1 = lm.to_linearizer(q_ind=[q1], qd_ind=[q1d], q_dep=[q2], qd_dep=[q2d])

# linearization about q1=0 & q2=l
op_point = {q1: 0, q1d: 0, q2: l, q2d: 0}

M1, A1, B1 = lm_lin1.linearize(op_point=op_point)

print(' ')
print('q1 is independent')
print('linear mass matrix:')
vprint(M1)
print('linear force matrix:')             
vprint(A1)
print('linear matrix B:')  
vprint(B1)

# linearization with q2 as independent variable
lm_lin2 = lm.to_linearizer(q_ind=[q2], qd_ind=[q2d], q_dep=[q1], qd_dep=[q1d])

# linearization about q1=lx & q2=ly
# note: linearization at q1=0 and q2=l gives a spurious result here (due to division by zero)
lx, ly = symbols('lx ly')
op_point2 = {q1: lx, q1d: 0, q2: ly, q2d: 0}

M2, A2, B2 = lm_lin2.linearize(op_point=op_point2)

print(' ')
print('q2 is independent')
print('linear mass matrix:')
vprint(M2)
print('linear force matrix:')             
vprint(A2)
print('linear matrix B:')  
vprint(B2)
