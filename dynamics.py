#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 21:03:23 2018

@author: julianehelbich
"""

# THIS FILE NEEDS THE FOLDER "./checkpoints" TO WORK! IT'S WHERE FILES GET STORED.


import numpy as np, math
from scipy.constants import g as GRAV, k as KB


# declare constants, variables etc.
EPSILON = 1.651e-21                 # energy scale in new unit system
MASS = 6.644e-26                    # mass scale in new unit system
SIGMA = 3.405e-10                   # length scale in new unit system
TAU = 4.756e-18                     # time scale in new unit system
numParticles = range(6)
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

#periodic and reflective boundary condition have to somehow be inlcuded in the code, 
#both should be working - different boxes, for reflective boundary condition the box is a part of a bigger volume, 
#therefor particles that are outside of our box have to be considert as well (rMax) -> different for the plot and animation        



# verlet takes position, velocities and force to compute next positions and velocities, returns new positions/velocities
# compute for different values of sigma for use in the diffusion coefficient module
for s in range(1,11):
    for i in range(4001):

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

        if i % 2000 == 0:                                             # save checkpoints for other modules to import
            np.savetxt('checkpoint_r_s=' + str(s) + '_i=' + str(i), initCoords)    #checkpoints are saved to user
            np.savetxt('checkpoint_v_s=' + str(s) + '_i=' + str(i), vMatrix)

#animated plot - positions 

import matplotlib.pyplot as plt
import matplotlib.animation as animation


        
#ANIMATION
#use ArtistAnimation class - animation using a fixed set of objects
#or use FuncAnimation - ania
#fig = plt.figure()


#BACKGROUND of animation: box - grid with 80 x 80 (boxlength)
#def box(self):
   # self.xlim = x0,x1 
    #self.ylim = y0,y1 
    #Y, X = np.mgrid[x0:x1:80, y0:y1:80]

    
#ax = plt.axes(Y, X)
#line = ax.plot([],[],)
#MOLECULES animation
#first:get positions for all particles from initCoords martix saved in checkpoints
class particles(object):
    for s in range(1,11):
        for i in range(4001):
            if i % 2000 == 0:      #loads all of the r checkpoints
                datacoord  = np.loadtxt('checkpoint_r_s=' + str(s) + '_i=' + str(i))
                #print('Coords of particles:' + str(s) + '_' + str(i))#print('datacoord') #print(datacoord) #prints every array for s,i
                data = np.hsplit(datacoord,2) #splits datacoord in two arrays column-wise
                #print('data')
                #print(data) #prints all split arrays for s,i
                #x and y coordinates for the particles from every position
                x = np.hstack((data[0])) #array for x position for all particles at one point in time 
                y = np.hstack((data[1]))
                for n in numParticles:
                    print(x[n])
    #def particleposition(self):
     #   self.x = x
      #  self.y = y
        
            #xpos(str(n)) = x[n]
            #xcoord = x_pos_(str(n))
            #print(xcoord)
    
    #def __init__():    #timed animation
        
        
   
               

        
#ani = animation.FuncAnimation(fig, particles, box)
#plt.show()