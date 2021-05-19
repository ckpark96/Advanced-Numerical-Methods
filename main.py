import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy.linalg as la
import math
# import sympy as sy

N = 300 # number of nodes in the domain
L = 10 # fixed
ubar = 1 # fixed

def f(x):
    res = np.zeros_like(x)
    res[(x>=2)&(x<=4)] = (1-np.cos(np.pi*x[(x>=2)&(x<=4)]))/2
    res[(x>=6)&(x<=8)] = 1
    return res

def Upwind_lin(IC,dx,dt,T): # first order upwind
    sol = np.zeros((IC.shape[0],int(T//dt)+1)) # rows: spatial, columns: temporal (+1 to account for t=0)
    sol[:,0] = IC # Add the initial condition

    for n in range(1,int(T//dt)+1): # progress in time but excluding t=0
        for i in range(sol.shape[0]): # progress in space
            sol[i,n] = sol[i,n-1]-ubar*dt/dx*(sol[i,n-1]-sol[i-1,n-1])

    return sol

def BeamWarming_lin(IC,dx,dt,T): # second order upwind
    sol = np.zeros((IC.shape[0],int(T//dt)+1))
    sol[:,0] = IC

    for n in range(0,int(T//dt)): 
        for i in range(sol.shape[0]): 
            sol[i,n+1] = sol[i,n]-ubar*dt/2/dx*(3*sol[i,n]-4*sol[i-1,n]+sol[i-2,n])+1/2*ubar**2*(dt/dx)**2*(sol[i,n]-2*sol[i-1,n]+sol[i-2,n]) # ubar is A (in the book)

    return sol


def BeamWarming_flux_limited_lin(IC,dx,dt,T,fluxlimiter):
    sol = np.zeros((IC.shape[0],int(T//dt)+1))
    sol[:,0] = IC
    N = IC.shape[0]
    nu = ubar*dt/dx
    for n in range(0,int(T//dt)):
        for i in range(sol.shape[0]):
            # print(sol[i+1,n]-sol[i,n])
            if np.abs(sol[(i+1)%N,n]-sol[i,n]) < 1e-10: # If sol is smooth, theta = 1
                thetaip = 1
            else:
                thetaip = (sol[i,n]-sol[i-1,n]) / (sol[(i+1)%N,n]-sol[i,n])
            if np.abs(sol[i,n]-sol[i-1,n]) < 1e-10:
                thetain = 1
            else:
                thetain = (sol[i-1,n]-sol[i-2,n]) / (sol[i,n]-sol[i-1,n])
            sol[i,n+1] = sol[i,n]-nu*(sol[i,n]-sol[i-1,n])-nu/2*(1-nu)*(fluxlimiter(thetaip)*(sol[(i+1)%N,n]-sol[i,n]) - fluxlimiter(thetain)*(sol[i,n]-sol[i-1,n]))

    return sol


def vanLeer(a): # flux limiter
    return (a+np.abs(a))/(1+np.abs(a))


dx = L/N
T = L*10/ubar
dt = 0.4*dx/ubar
dtpT = L/ubar/dt
x = np.linspace(dx/2,L-dx/2,N) # Uniform grid
xnon = np.concatenate((np.linspace(dx/2,L/2,N-(int(N/3))), np.linspace(L/2, L-dx/2,int(N/3)))) # Non-uniform grid
IC = f(x)*dx

# solution = BeamWarming_lin(IC,dx,dt,T)/dx
# plt.plot(solution[:,-1],label="Beam Warming")
# solution = Upwind_lin(IC,dx,dt,T)/dx
# plt.plot(solution[:,-1],label="Upwinding")
# solution = BeamWarming_flux_limited_lin(IC,dx,dt,T,vanLeer)/dx
# plt.plot(solution[:,-1],label="van Leer")
# plt.plot(solution[:,0],"k--",label="Exact")
# plt.legend()
# plt.show()


''' Acoustic '''
A = np.array([[0, -4],[-1,0]])
eigval, eigvec = la.eig(A) # eigenvec are wrong
lam_neg = eigval[1]
lam_pos = eigval[0]
Z = 2
eigvec1 = np.array([[-Z],[1]])
eigvec2 = np.array([[Z],[1]])
R = np.hstack((eigvec1,eigvec2))


def alpha1(p0,p1,u0,u1,Z):
    return 1/(2*Z)*(-1*(p1-p0)+Z*(u1-u0))

def alpha2(p0,p1,u0,u1,Z):
    return 1/(2*Z)*((p1-p0)+Z*(u1-u0))    

def godunov_acous(ICp,ICu,dx,dt,T,Z,lam1,lam2): # first order upwind
    N = ICp.shape[0]
    r1 = np.array([[-Z],[1]])
    r2 = np.array([[Z],[1]])
    sol_p = np.zeros((IC.shape[0],int(T//dt)+1)) # along rows: spatial adv, along columns: temporal adv (+1 to account for t=0)
    sol_u = np.zeros((IC.shape[0],int(T//dt)+1))
    sol_p[:,0] = ICp # Add the initial condition
    sol_u[:,0] = ICu

    for n in range(1,int(T//dt)+1): # progress in time but excluding t=0
        for i in range(sol_p.shape[0]): # progress in space
            
            alph1n = alpha1(sol_p[i-1,n-1],sol_p[i,n-1],sol_u[i-1,n-1],sol_u[i,n-1],Z) # alpha^1_i-0.5
            alph2n = alpha2(sol_p[i-1,n-1],sol_p[i,n-1],sol_u[i-1,n-1],sol_u[i,n-1],Z) # alpha^2_i-0.5
            alph1p = alpha1(sol_p[i,n-1],sol_p[(i+1)%N,n-1],sol_u[i,n-1],sol_u[(i+1)%N,n-1],Z) # alpha^1_i+0.5
            alph2p = alpha2(sol_p[i,n-1],sol_p[(i+1)%N,n-1],sol_u[i,n-1],sol_u[(i+1)%N,n-1],Z) # alpha^2_i+0.5

            Qprev = np.array([[sol_p[i,n-1]],[sol_u[i,n-1]]]) # Q^n_i
            # Q = Qprev - dt/dx*(2*alph1n*r1 - 2*alph2p*r2)
            Q = Qprev - dt/dx*(2*alph2n*r2 - 2*alph1p*r1)
            sol_p[i,n] = Q[0,:]
            sol_u[i,n] = Q[1,:]
           
    return sol_p, sol_u

def BeamWarming_acou(ICp,ICu,dx,dt,T,lam1,lam2): # second order upwind
    A = np.array([[lam1, 0],[0, lam2]])
    sol_p = np.zeros((IC.shape[0],int(T//dt)+1)) # along rows: spatial adv, along columns: temporal adv (+1 to account for t=0)
    sol_u = np.zeros((IC.shape[0],int(T//dt)+1))
    sol_p[:,0] = ICp # Add the initial condition
    sol_u[:,0] = ICu

    for n in range(1,int(T//dt)+1): 
        for i in range(sol_p.shape[0]):
            Qprev = np.array([[sol_p[i,n-1]],[sol_u[i,n-1]]])
            Qprevprev = np.array([[sol_p[i-1,n-1]],[sol_u[i-1,n-1]]])
            Qprevprevprev = np.array([[sol_p[i-2,n-1]],[sol_u[i-2,n-1]]])
            first = 3*Qprev - 4*Qprevprev + Qprevprevprev
            second =  Qprev - 2*Qprevprev + Qprevprevprev
            Q = Qprev - dt/2/dx*A@first + 0.5*(dt/dx)**2*A**2@second
            sol_p[i,n] = Q[0,:]
            sol_u[i,n] = Q[1,:]

    return sol_p, sol_u

ICp = f(x)
ICu = np.ones_like(ICp)

sol_p_god, sol_u_god = godunov_acous(ICp,ICu,dx,dt,T,Z,lam_pos,lam_neg)
sol_p_beam, sol_u_beam = BeamWarming_acou(ICp,ICu,dx,dt,T,lam_pos,lam_neg)

plt.figure()
plt.title('p')
plt.plot(sol_p_god[:,-1],label="Godunov")
# plt.plot(sol_p_beam[:,50],label="Beam-Warming")
plt.plot(sol_p_beam[:,-1],label="Beam-Warming")
# plt.plot(sol_p[:,0],'k--',label='Exact')
plt.legend()
plt.figure()
plt.title('u')
plt.plot(sol_u_god[:,-1],label="Godunov")
# plt.plot(sol_u_beam[:,50],label="Beam-Warming")
plt.plot(sol_u_beam[:,-1],label="Beam-Warming")
# plt.plot(sol_u[:,0],'k--',label='Exact')

plt.show()
