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
# Set up for Animation

## System Parameters
A = np.array([[0.906488,0.0816012,-0.0005],[0.0741349,0.90121,-0.000708383],[0,0,0.132655]])
B = np.array([[-0.00150808,-0.0096,0.867345]]).T
D = np.array([[0.00951892,0.00038373,0]]).T

Q = np.diag([1,1,1])
R=1
gamma = 5 
##

n=np.shape(A)[1];
m1=np.shape(B)[1];
m2=np.shape(D)[1];
omega_0 = 1;
l_a =50;
## Initial control matrix
P0 = np.zeros((n,n))

# Stores the control matrix

P = [P0]
##
n_end = 40
n_learn =150
x = np.zeros((n,n_end))
u = np.zeros((m1,n_end))
w = np.zeros((m2,n_end))
x_0  = np.array([[1], [-1], [-1]])
x[:,0] = x_0[:,0]

for i in range(n_learn):
    Phi = np.zeros((n_end,n*n))
    Psi = np.zeros((n_end,1))
    if i > l_a-1:
        omega = 1 
    else:
        omega = omega_0    
    for k in range(n_end-1): # Collect Data
        u[:,k]=-(1/2)*(R**(-1))*B.T@P[i]@A@ x[:, k]
        w[:,k] = (1/2)*(gamma**(-2))*D.T@P[i]@A@ x[:, k]
        x[:, k+1] = A @ x[:, k] + B @ u[:, k] + D @ w[:, k] 
        Phi[k, :] = np.kron(x[:, k].T, x[:, k].T)
        Psi[k] = omega*( x[:, k].T @ Q @ x[:, k] + u[:, k].T * R * u[:, k] - gamma**2 * w[:, k].T * w[:, k]+x[:, k+1].T @ P[i] @ x[:, k+1]) + (1-omega)*x[:, k].T @ P[i] @ x[:, k]
    vec_P = np.linalg.pinv(np.transpose(Phi) @ Phi) @ np.transpose(Phi) @ Psi
    P.append(np.reshape(vec_P, (n, n))) # Find P

dP = np.zeros((n_learn,1))

for j in range(n_learn):
    dP[j] = np.linalg.norm(P[j])

with open('Case_2.npy', 'wb') as f:
    np.save(f,dP)
    
# # =============================================================================

# t = np.arange(n_learn)
# t = t.reshape(1, -1)
# fig, ax = plt.subplots(facecolor='white')
# ax.set_facecolor('white')
# ax.clear()
# ax.plot(t, dP.T, "-o",color='cyan')
# ax.grid()
# ax.set(title="Iteration: ", xlim=(-0.5, n_learn), ylim=(-0.1, 30)) 

# # =============================================================================

