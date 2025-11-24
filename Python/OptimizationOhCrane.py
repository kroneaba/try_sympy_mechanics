# Optimization of an overhead crane
#
# object function: minimum mass of girder and hoist
#
# variables: bG, tG, hS, tS (i.e. dimensions of the tee section)
#
from scipy.optimize import minimize

from OverheadCrane import *

#
# --------------------------------------------------------------------------------------
#
# optimization variables
#
# initial guess
#
bG = 350.
tG = 22.
hS = 200.
tS = 16.
#
x0 = [bG, tG, hS, tS] # array of variables
#
parameters_vals = [mLift, L, rho] # parameters needed for objective function
arguments = (parameters_vals,) # tuple with parameters 

# define bounds
b_a1 = (bGmin, bGmax)
b_a2 = (tGmin, tmax)
b_a3 = (hSmin, hSmax)
b_a4 = (tmin, tmax)
bnds = (b_a1, b_a2, b_a3, b_a4)
#
# --------------------------------------------------------------------------------------
#
# objective function
#
def totalMass(x, args):
# x: optimization variables
# args: geometric parameters
#
# returns total weight of girder and hoist
# devided by mL (as scale factor)
    mtot = massHoist(x[0])
    Atot, dum1, dum2, dum3, dum4 = section(x[0], x[1], x[2], x[3])
    mtot += args[2] * Atot * args[1]
    return mtot/args[0]
#
# --------------------------------------------------------------------------------------
# define optimization constraint
# violation of inequality constraints if < 0
#
# utilization due to maximum tensile stress
def constraintTensStress(x, args):
    Atot, As, Itot, et, ec = section(x[0], x[1], x[2], x[3])
    Fw, qL, Fmax, Mmax = maxForces(x[0], Atot)
    constraintVal = utilTensStress(x[1], Mmax, Itot, et)
    return 1.-constraintVal
#
# utilization due to maximum compressive stress
def constraintCompStress(x, args):
    Atot, As, Itot, et, ec = section(x[0], x[1], x[2], x[3])
    Fw, qL, Fmax, Mmax = maxForces(x[0], Atot)
    constraintVal =utilCompStress(x[3], Mmax, Itot, ec)
    return 1.-constraintVal
#
# utilization due to maximum shear stress
def constraintShearStress(x, args):
    Atot, As, Itot, et, ec = section(x[0], x[1], x[2], x[3])
    Fw, qL, Fmax, Mmax = maxForces(x[0], Atot)
    constraintVal = utilSheartress(x[3], Fmax, As)
    return 1.-constraintVal
#
# utilization due to maximum deflection
def constraintMaxDef(x, args):
    Atot, As, Itot, et, ec = section(x[0], x[1], x[2], x[3])
    Fw, qL, Fmax, Mmax = maxForces(x[0], Atot)
    constraintVal = utilMaxW(Fw, qL, Itot)
    return 1.-constraintVal
#
# utilization due to minimum thickness of flange
def constraintMinFlange(x, args):
    constraintVal = utilMin_tG(x[0], x[1])
    return 1.-constraintVal
#
# utilization due to thickness ratio 1
def constraintTRatio1(x, args):
    constraintVal = utilThicknessRatio(x[1], x[3])
    return 1.-constraintVal
#
# utilization due to thickness ratio 2
def constraintTRatio2(x, args):
    constraintVal = utilThicknessRatio(x[3], x[1])
    return 1.-constraintVal
#
# utilization due to permissible height
def constraintMaxHeight(x, args):
    constraintVal = utilMaxHeight(x[1], x[2])
    return 1.-constraintVal
#
cons = []  # list with constraints
#
cons_i = ({'type':'ineq', 'fun':constraintTensStress, 'args':arguments})
cons.append(cons_i)
#
cons_i = ({'type':'ineq', 'fun':constraintCompStress, 'args':arguments})
cons.append(cons_i)
#
cons_i = ({'type':'ineq', 'fun':constraintShearStress, 'args':arguments})
cons.append(cons_i)
#
cons_i = ({'type':'ineq', 'fun':constraintMaxDef, 'args':arguments})
cons.append(cons_i)
#
cons_i = ({'type':'ineq', 'fun':constraintMinFlange, 'args':arguments})
cons.append(cons_i)
#
cons_i = ({'type':'ineq', 'fun':constraintTRatio1, 'args':arguments})
cons.append(cons_i)
#
cons_i = ({'type':'ineq', 'fun':constraintTRatio2, 'args':arguments})
cons.append(cons_i)
#
cons_i = ({'type':'ineq', 'fun':constraintMaxHeight, 'args':arguments})
cons.append(cons_i)
#
# --------------------------------------------------------------------------------------
# results of initial configuration
#
showResults(x0[0], x0[1], x0[2], x0[3], title='initial configuration')
#
# --------------------------------------------------------------------------------------
# perform optimization 
#
# method SLSQP: Sequential Least Squares Programming
#
sol = minimize(totalMass, x0, args=arguments, method='trust-constr', \
               bounds=bnds, constraints=cons, options={'disp':True})
print(' ')
print(sol)
print(' ')
print(sol.values())
print(' ')
print(sol.x)
print(' ')
#
# --------------------------------------------------------------------------------------
# optimal configuration
#
showResults(sol.x[0], sol.x[1], sol.x[2], sol.x[3], title='optimal configuration')


