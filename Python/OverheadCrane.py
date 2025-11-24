from numpy import sqrt
import matplotlib.pyplot as plt
#
# --------------------------------------------------------------------------------------
#
# data of overhead crane
# units:
# length: mm
# mass: kg
# force: N
# stress: MPa or N/mm^2
#
mLift = 10000.  # lift capacity
L = 6000.  # span
mK0 = 2000.  # basic mass of hoist
aW = 400.  # minimum distance from wall
Hmax = 600.  # maximum height of girder
#
# material
E = 210000.  # elasticity modulus
rho = 7.85E-6  # density
Reh = 355.  # yield strength, t<=16mm
Reh2 = 345.  # yield strength, 16mm<t<=40mm
tlim = 16.  # threshold thickness for yield strength
jstens = 1.25  # safety factor - tension (and shear)
jscomp = 1.1  # safety factor - compression
#
# general boundaries
tmin = 5.  # minimum plate thickness
tGmin = 16.  # minimumthickness of flange
tmax = 40. # maximum plate thickness
bGmin = 300.0  # minimum width of flange
bGmax = 600.0  # maximum width of flange
hSmin = 100.0  # minimum height of web
hSmax = Hmax  # maximum height of web
#
# additional constants
g = 9.81  # gravitational acceleration in m/s^2
fL = 1.5  # load factor (roughly covers dynamical effects at lifting
#
# --------------------------------------------------------------------------------------
#
# functions
#
# mass of hoist
#
def massHoist(b):
#
    mHoist = mK0
    if b > bGmin:
        mHoist += 0.1*mHoist*(b-bGmin)
#
    return mHoist
#
#  properties of tee section
#
def section(bG, tG, hS, tS):
#
    A_G = bG*tG
    A_S = hS*tS
    Atot = A_G + A_S
    yC = (A_G*tG/2. + A_S*(tG+hS/2.)) / Atot  # location of centroid acc. to Roark
    I_G = (bG*tG**3)/12.
    I_S = (tS*hS**3)/12.
    Itot = I_G + A_G*(yC-tG/2.)**2 + I_S + A_S*(tG+hS/2.-yC)**2
    ed = tG+hS-yC
#
    return Atot, A_S, Itot, yC, ed
#
#  maximum bending moment and shear force
#
def maxForces(bG, A):
#
    mK = massHoist(bG)  # actual mass of hoist
    Fw = (mK+mLift*fL)*g   # weight force of lifting mass and hoist
    qL = rho*A*g*L  # force due to girder weight
    Mmax = Fw*L/4.  # bending moment from weight forces
    Mmax += qL*L/8.  # bending moment from girder weight force added
    Fmax = qL/2. +(L-aW)*Fw/L  # shear force due to weight forces added
#
    return Fw, qL, Fmax, Mmax
#
#  maximum deflection
#
def maxW(Fw, qL, I):
#
    wmax = Fw*L**3/(48.*E*I)  # portion due to weight force of lifting mass and hoist
    wmax += 5.*qL*L**3/(384.*E*I)
#
    return wmax
#
#
def utilTensStress(tG, Mmax, I, et):
# utilization tensile stress
    RehG = Reh
    if (tG>tlim):
        RehG = Reh2
    sigma_t = Mmax/I*et
    return jstens*sigma_t/RehG
#
def utilCompStress(tS, Mmax, I, ec):
# utilization compressive stress (amouunt)
    RehS = Reh
    if (tS>tlim):
        RehS = Reh2
    sigma_c = Mmax/I*ec
    return jscomp*sigma_c/RehS
#
def utilSheartress(tS, Fmax, As):
# utilization shear stress (simplified)
    RehS = Reh
    if (tS>tlim):
        RehS = Reh2
    tau = Fmax/As
    return jstens*tau/RehS*sqrt(3.)
#
def utilMaxW(Fw, qL, I):
# utilization maximum deflection
    wmax = maxW(Fw, qL, I)
    return wmax*500./L
#
def utilMin_tG(bG, tG):
# utilization minimum thickness of flange
    return tGmin*(bG/bGmin)**2/tG
#
def utilThicknessRatio(t1, t2):
# utilization thickness ratio flange and web
    return t1/(t2*1.6)
#
def utilMaxHeight(tG, hS):
# utilization of max. height of section
    return (tG+hS)/Hmax
#
# --------------------------------------------------------------------------------------
#
# plot-routines
#
# plot section
#
def plotSection(bG, tG, hS, tS, yC, Atot, As, I, title='tee-section'):
# polygon
    xs = [bG/2., bG/2., tS/2., tS/2., -tS/2., -tS/2., -bG/2., -bG/2., bG/2.]
    ys = [0., tG, tG, tG+hS, tG+hS, tG, tG, 0., 0.]
#
    plt.title(title)
#
    plt.fill(xs, ys) # section
    plt.plot([0.], [yC], 'ko')
    xmin = -0.55*Hmax
    xmax = 0.55*Hmax
    ymin = -0.05*Hmax
    ymax = 1.05*Hmax
    plt.axis((xmin, xmax, ymin, ymax))
    plt.grid()
#
    plt.text(0.25*Hmax, 0.9*Hmax, 'Atot='+str(int(Atot)), backgroundcolor='w')
    plt.text(0.25*Hmax, 0.8*Hmax, 'As='+str(int(As)), backgroundcolor='w')
    plt.text(0.25*Hmax, 0.7*Hmax, 'Ix='+str(int(I)), backgroundcolor='w')
#
    plt.show()
#
    return
#
# plot girder
#
def plotGirder(bG, tG, hS, tS, Atot, title='girder'):
# polygons
    xs = [bG/2., bG/2., tS/2., tS/2., -tS/2., -tS/2., -bG/2., -bG/2., bG/2.]
    ys = [0., tG, tG, tG+hS, tG+hS, tG, tG, 0., 0.]
    xg = [L/2., L/2., -L/2., -L/2., L/2]
    htot = tG+hS
    yg = [0., htot, htot, 0., 0.]
    xg2 = [-L/2., L/2.]
    yg2 = [tG, tG]
#
    plt.title(title)
#
    plt.plot(xg, yg, color='k', linewidth=2)
    plt.plot(xg2, yg2, color='k', linewidth=2)
    plt.plot(xs, ys, color='k', linewidth=1, linestyle='dashed')
#
    plt.axis('equal')
    plt.grid()
#    
    plt.text(0.3*L, 0.35*L, 'htot='+str(int((tG+hS)*10)/10), backgroundcolor='w')
    plt.text(0.3*L, 0.30*L, 'bG='+str(int(bG*10)/10), backgroundcolor='w')
    mG = rho*Atot*L
    plt.text(0.3*L, 0.25*L, 'mG='+str(int(mG*10)/10), backgroundcolor='w')
    plt.show()
#
    return
#
# output for a given configuration
#
def showResults(bG, tG, hS, tS, title='test'):
    #
    Atot, A_S, Itot, et, ec = section(bG, tG, hS, tS)
#
    Fw, qL, Fmax, Mmax = maxForces(bG, Atot)
#
    wmax = maxW(Fw, qL, Itot)
#
    print(' ')
    print('actual results for '+title+':')
    print('weight force =', Fw)
    print('max. force =', Fmax)
    print('max.bending moment =', Mmax)
    print('max. bending deflection =', wmax)
    print('utilizations:')
    print('tensile stress: ', utilTensStress(tG, Mmax, Itot, et))
    print('compressive stress: ', utilCompStress(tS, Mmax, Itot, ec))
    print('shear stress: ', utilSheartress(tS, Fmax, A_S))
    print('max. deflection: ', utilMaxW(Fw, qL, Itot))
    print('min. thickness flange: ', utilMin_tG(bG, tG))
    print('thickness ratio 1: ', utilThicknessRatio(tG, tS))
    print('thickness ratio 2: ', utilThicknessRatio(tS, tG))
    print('max. height: ', utilMaxHeight(tG, hS))
    print(' ')
#
    plotSection(bG, tG, hS, tS, et, Atot, A_S, Itot, title)
#
    plotGirder(bG, tG, hS, tS, Atot, title)
#
    return

#
# --------------------------------------------------------------------------------------
#
# main
#
if __name__ == "__main__":
#
# geometry
    bG = 300.
    tG = 25.
    hS = 360.
    tS = 35.
#
# show results of this configuration
#
    showResults(bG, tG, hS, tS, title='test configuration')
