# -*- coding: utf-8 -*-
"""
Created on Wed May 24 2021

@author: Maywell2019
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib.colors import Normalize
from matplotlib import cm

def getE(S, L1, L2, L3):
    S11 = S[0,0]
    S12 = S[0,1]
    S44 = S[3,3]
    A = 2 * (S11 - S12) / S44
    E = S11 + (1-A) * S44 * ( (L1*L2)**2 + (L2*L3)**2 + (L3*L1)**2 )
    E = 1 / E
    Emin = np.min(E)
    Emax = np.max(E)
    return E,A,Emin,Emax

def plot_ansiotropy(Name,C11,C12,C44):
    C = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            if i < 3 and j < 3 and j == i:
                C[i,j] = C11
            if i < 3 and j < 3 and j != i:
                C[i,j] = C12
            if i > 2 and j == i:
                C[i,j] = C44
    
    S = np.linalg.inv(C)
    
    theta, phi = np.linspace(0, np.pi, 200), np.linspace(0, 2 * np.pi, 200)
    THETA, PHI = np.meshgrid(theta, phi)
    L1 = np.sin(THETA) * np.cos(PHI)
    L2 = np.sin(THETA) * np.sin(PHI)
    L3 = np.cos(THETA)
    
    [E,A,Emin,Emax] = getE(S, L1, L2, L3)
    
    X = E*L1
    Y = E*L2
    Z = E*L3
    
    color_map = plt.get_cmap('rainbow')
    scalarmap = cm.ScalarMappable(norm=Normalize(vmin=Emin,vmax=Emax),cmap=color_map)
    C_colored = scalarmap.to_rgba(E)
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 28,
    }
    
    fig = plt.figure(figsize=(15,12))
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.view_init(elev=30,azim=45)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,facecolors=C_colored, linewidth=0, antialiased=False)
    #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap=cm.RdYlGn, linewidth=0, antialiased=False)
    #ax.plot_surface(X, Y, Z, rstride=8, cstride=8,alpha=0.3)
    ax.contour(X, Y, Z, zdir='x', offset=-175)
    ax.contour(X, Y, Z, zdir='y', offset=-175)
    ax.contour(X, Y, Z, zdir='z', offset=-175)
    cbar = plt.colorbar(scalarmap,shrink=0.6,aspect=15)
    cbar.set_label('GPa')
    plt.title(Name,font1)
    fig_name = Name + '.png'
    plt.savefig(fig_name,bbox_inches = 'tight',dpi=300)
#    plt.show()

dataset = pd.read_csv('Total_results.csv')

for index, row in dataset.head(5).iterrows():
#    print(row)
#    print(row['C11'])
    plot_ansiotropy(row['Alloy'],row['C11'],row['C12'],row['C44'])
    