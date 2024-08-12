# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:16:49 2022

@author: daimi
"""

import numpy as np
from pyevtk.hl import gridToVTK


def readfile(Nx, Ny, Nz,step):
    filename = 'grainvol%d.txt' % step
    indata = np.loadtxt(filename, skiprows = 1)
    grainvol = np.zeros((Nx, Ny, Nz))
    for i in range(len(indata)):
        grainvol[int(indata[i,0]), int(indata[i, 1]), int(indata[i,2])] = indata[i, 3]
    
    return grainvol


def outputvtk(grainvol, dataid, Nx, Ny, Nz):
    # specify location
    x = np.zeros((Nx, Ny, Nz))
    y = np.zeros((Nx, Ny, Nz))
    z = np.zeros((Nx, Ny, Nz))
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                x[i,j,k] = i 
                y[i,j,k] = j 
                z[i,j,k] = k
    # specify filename
    filename = "./grainvol_%08d" % dataid
    gridToVTK(filename, x, y, z, pointData = {"grainvol": grainvol})
    
Nx = 100; Ny = 100; Nz = 100
Nstep = 1; Noutput = 1
for i in range(0, Nstep + Noutput, Noutput):
    grainvol=readfile(Nx, Ny, Nz, i)
    outputvtk(grainvol,i,Nx,Ny,Nz)

# grainvol=readfile(Nx, Ny, Nz,0)
# outputvtk(grainvol,0,Nx,Ny,Nz)