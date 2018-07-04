# THIS FILE NEEDS THE FOLDER "./checkpoints" TO WORK! IT'S WHERE FILES GET STORED.


import numpy as np, math
from scipy.constants import g as GRAV, k as KB

# declare constants, variables etc.
EPSILON = 1.651e-21                 # energy scale in new unit system
MASS = 6.644e-26                    # mass scale in new unit system
SIGMA = 3.405e-10                   # length scale in new unit system
TAU = 4.756e-18                     # time scale in new unit system
numParticles = 100
boxLength = 80                      # length of simulated space
TEMP = 288.15
VAR = math.sqrt(KB*TEMP/EPSILON)    # variance of particle speed according to ideal gas theory
rMax = 0.2*boxLength                # cutoff range for Leonard-Jones-Potential
dt=2e-3                             # time stepping for integration

# get initial coordinates, randomly distributed in space
initCoords = np.random.sample((numParticles,2))*boxLength

# create array with distances
rMatrix = np.tile(initCoords,(numParticles,1)).reshape(numParticles,numParticles,2) # create 100x100x2 from 100x2
rMatrix -= np.transpose(rMatrix,(1,0,2))        # transpose and substract to create all possible pairs of distances

# create velocity distribution, dx and dy normal distributed -> [dx,dy] maxwell-boltzmann distributed
initVel0 = []
dxList = np.random.normal(0,VAR,100)
dyList = np.random.normal(0,VAR,100)
for i in range(numParticles):
    initVel0.append([dxList[i],dyList[i]])

# sum over all velocities to get net total velocity of all particles
netTotalVel = [0,0]
for i in range(len(initVel0)):
    for j in range(len(initVel0[i])):
        netTotalVel[j] += initVel0[i][j]

# substract from each velocity to retain distribution and get zero net momentum.
vMatrix = np.array(initVel0)
for i in range(len(initVel0)):
    for j in range(len(netTotalVel)):
        vMatrix[i][j] -= netTotalVel[j]/len(initVel0)


# define function that computes force from leonard-jones-potential, gets called by verlet.py
def rAbs(r):
    return np.sqrt(r[0]**2+r[1]**2)

def forceLJ(r, s):
    if rAbs(r) < 1e-20:
        return np.array([0.,0.])
    else:
        return -4*(12*s**12/rAbs(r)**14 - 6*s**6/rAbs(r)**8)*np.array(r)

def getForce(r, s):
    r = np.array(r)
    if rAbs(r) < rMax:                                                  #try withouht periodic boundary condition
        return forceLJ(r, s)
    r -= np.array([np.sign(r[0])*boxLength,0])                          #try first periodic boundary condition
    if rAbs(r) < rMax:
        return forceLJ(r, s)
    r -= np.array([np.sign(r[0])*boxLength,np.sign(r[1])*boxLength])    # try second periodic boundary condition
    if rAbs(r) < rMax:
        return forceLJ(r, s)
    else:
        return np.array([0.,0.])



# verlet takes position, velocities and force to compute next positions and velocities, returns new positions/velocities
# compute for different values of sigma for use in the diffusion coefficient module
for s in range(1,10):
    for i in range(4000):

        fMatrix = np.apply_along_axis(getForce, 2, rMatrix, s)      # get pairwise forces, matrix 100x100x2 at time=t
        force = np.sum(fMatrix, 0)                                  # get net force per particle, vector 100x2 at time=t

        vMatrixDeltaT = vMatrix + force*dt*0.5                      # speed vector for t=t+dt/2, 100x2
        initCoords += vMatrixDeltaT*dt                              # position vector for t+dt, 100x2

        # create array with distances at t+dt/2
        rMatrix = np.tile(initCoords,(numParticles,1)).reshape(numParticles,numParticles,2) # create 100x100x2 from 100x2
        rMatrix -= np.transpose(rMatrix,(1,0,2))                    # transpose and substract to create all possible pairs of distances

        fMatrix = np.apply_along_axis(getForce, 2, rMatrix, s)      # get pairwise forces, matrix 100x100x2, t+dt
        force = np.sum(fMatrix, 0)                                  # get net force per particle, vector 100x2, t+dt

        vMatrix = vMatrixDeltaT + force*dt*0.5                      # new speed vector at t+dt

        initCoords = np.where(initCoords < 0, initCoords % boxLength, initCoords)           # apply lower/left boundary conditions
        initCoords = np.where(initCoords > boxLength, initCoords % boxLength, initCoords)   # apply upper/right boundary conditions

        if i % 20 == 0:                                             # save checkpoints for other modules to import
            np.save('./checkpoints/EDLr%s_%i.npy' %(s, i), initCoords)
            np.save('./checkpoints/EDLv%s_%i.npy' %(s, i), vMatrix)
