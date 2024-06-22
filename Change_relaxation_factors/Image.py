# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:53:18 2024
Code for paper: Accelerated Value Iteration for Nonlinear Zero-Sum Games
with Convergence Guarantee.
               
Programing Language : Python
Purpose : Practice and Research

@author: Admin
"""
## Import Lib
import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.animation as animation
import math
plt.style.use("ggplot")
plt.rcParams["figure.dpi"]= 200
plt.rcParams["figure.figsize"]= (10,6)
plt.rcParams["figure.constrained_layout.use"]= False
# plt.rcParams['text.usetex'] = False
# Set up for Animation

with open('Case_1.npy', 'rb') as f1:
    dP_c1 = np.load(f1)
    
with open('Case_2.npy', 'rb') as f2:
    dP_c2 = np.load(f2)

with open('Case_3.npy', 'rb') as f3:
    dP_c3 = np.load(f3)

with open('Case_4.npy', 'rb') as f4:
    dP_c4 = np.load(f4)    
    
# =============================================================================
n_learn = 150
t = np.arange(n_learn)
fig, ax = plt.subplots(facecolor='white')
ax.set_facecolor('white')
ax.clear()
ax.plot(t, dP_c1[:, 0], '-', color='cyan')
ax.plot(t, dP_c2[:, 0], "-",color='lime')
ax.plot(t, dP_c3[:, 0], "-",color='red')
ax.plot(t, dP_c4[:, 0], "-",color='fuchsia')
ax.set_xlabel(r'Iteration')
ax.set_ylabel(r'$ \| P^i \| $', fontsize=16)
ax.set_title(r'Accelerated Value Iteration', fontsize=16, color='deepskyblue')
ax.legend(["Case 1", "Case 2","Case 3", "Case 4"], loc="lower right")
ax.grid(color = 'green', linestyle = '--', linewidth = 0.5)
ax.set(xlim=(-0.5, n_learn), ylim=(-0.1, 30)) 
plt.savefig('Change_relaxation_factors.png')
# =============================================================================