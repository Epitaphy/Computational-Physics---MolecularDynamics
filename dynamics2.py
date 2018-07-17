#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 21:40:26 2018

@author: julianehelbich
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np, math
import scipy.integrate as integrate
from scipy.spatial.distance import pdist, squareform
from scipy.constants import g as GRAV, k as KB

# declare constants, variables etc.
EPSILON = 1.651e-21                 # energy scale in new unit system
MASS = 6.644e-26                    # mass scale in new unit system
SIGMA = 3.405e-10                   # length scale in new unit system
TAU = 4.756e-18                     # time scale in new unit system
numParticles = 5
boxLength = 100                     # length of simulated space
TEMP = 288.15
VAR = math.sqrt(KB*TEMP/EPSILON)    # variance of particle speed according to ideal gas theory
rMax = 0.2*boxLength                # cutoff range for Leonard-Jones-Potential
dt=2e-3                             # time stepping for integration

class MolecularDynamics:
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

#periodic and reflective boundary condition have to somehow be inlcuded in the code, 
#both should be working - different boxes, for reflective boundary condition the box is a part of a bigger volume, 
#therefor particles that are outside of our box have to be considert as well (rMax) -> different for the plot and animation        



# verlet takes position, velocities and force to compute next positions and velocities, returns new positions/velocities
# compute for different values of sigma for use in the diffusion coefficient module
   
    for s in range(1,11):
        for i in range(4001):

            fMatrix = np.apply_along_axis(getForce, 2, rMatrix, s)      
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
                
            Coords = initCoords
                
                
    #def initial state
    def __init__(self, initCoords, Coords, bounds = [0, boxLength, 0, boxLength], mass = 0.05, size = 0.1):
        self_initCoords = np.asarray(initCoords, dtype=float)
        self.mass = mass
        self.size = size
        self.Coords = Coords
        self.time_elapsed = 0
        self.bounds = bounds
        
    def step(self, dt):
        self.time_elapsed += dt
        self.state[:, :2] += dt * self.Coords[:, :2]   #for updating position

#set up initial state of molecules
#np.random.seed(0)
#initCoords = 

dynamics = MolecularDynamics(initCoords, size=0.1)     
        
fig = plt.figure
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(0, boxLength), ylim=(0, boxLength))
molecules, = ax.plot ([], [], 'bo', ms=6)

def init():
    global dynamics
    molecules.set_data([], [])
    return partciles

def animate(n):
    global dynamics, dt, ax, fig
    dynamics.step(dt)
    ms = int(fig.dpi * 2 * dynamics.size * fig.get_figwidth() / np.diff(ax.get_xbound())[0])
    molecules.set_data(dynamics.state[:, 0], dynamics.state[:, 1])
    molecules.set_markersize(ms)
    return molecules

ani = animation.FuncAnimation(fig, animate, frames=600, interval=10, blit=True, init_func=init)
plt.show