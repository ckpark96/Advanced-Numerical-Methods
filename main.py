import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy.linalg as la
# import sympy as sy

N = 300 # number of nodes in the domain
L = 10 # fixed
ubar = 1 # fixed

def f(x):
    res = np.zeros_like(x)
    res[(x>=2)&(x<=4)] = (1-np.cos(np.pi*x[(x>=2)&(x<=4)]))/2
    res[(x>=6)&(x<=8)] = 1
    return res

def Upwind_lin(IC,dx,dt,T,ubar=ubar): # first order upwind
    sol = np.zeros((IC.shape[0],int(T//dt)+1)) # rows: spatial, columns: temporal (+1 to account for t=0)
    sol[:,0] = IC # Add the initial condition

    for n in range(1,int(T//dt)+1): # progress in time but excluding t=0
        for i in range(sol.shape[0]): # progress in space
            sol[i,n] = sol[i,n-1]-ubar*dt/dx*(sol[i,n-1]-sol[i-1,n-1])

    return sol

def BeamWarming_lin(IC,dx,dt,T,ubar=ubar): # second order upwind
    sol = np.zeros((IC.shape[0],int(T//dt)+1))
    sol[:,0] = IC

    for n in range(0,int(T//dt)): 
        for i in range(sol.shape[0]): 
            sol[i,n+1] = sol[i,n]-ubar*dt/2/dx*(3*sol[i,n]-4*sol[i-1,n]+sol[i-2,n])+1/2*ubar**2*(dt/dx)**2*(sol[i,n]-2*sol[i-1,n]+sol[i-2,n]) # ubar is A (in the book)

    return sol


def BeamWarming_flux_limited_lin(IC,dx,dt,T,fluxlimiter,ubar=ubar):
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
dt = 0.9*dx/ubar
dtpT = L/ubar/dt
x = np.linspace(dx/2,L-dx/2,N) # Uniform grid
xnon = np.concatenate((np.linspace(dx/2,L/2,N-(int(N/3))), np.linspace(L/2, L-dx/2,int(N/3)))) # Non-uniform grid
IC = f(x)*dx
# print(IC.shape)
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
eigval = np.sort(eigval)
eigvec = eigvec[:,::-1]
Rinv = np.linalg.inv(eigvec)

def decompose(state):
    # (2,N) shape
    return Rinv@state

def recompose(state):
    # (2,N) shape
    return eigvec@state

def plus(x):
    return max(0,x)

def minus(x):
    return min(0,x)


def godunov_acous(IC,dx,dt,T, lambd): # first order upwind
    N = IC.shape[1]
    sol = np.zeros((2,N,int(T//dt)+1)) # rows: spatial, columns: temporal (+1 to account for t=0)
    sol[:,:,0] = decompose(IC)

    for n in range(1,int(T//dt)+1): # progress in time but excluding t=0
        for i in range(sol.shape[1]): # progress in space
            for j in range(lambd.shape[0]):
                sol[j,i,n] = sol[j,i,n-1] - dt/dx*(plus(lambd[j])*(sol[j,(i)%N,n-1]-sol[j,(i-1)%N,n-1])+minus(lambd[j])*(sol[j,(i+1)%N,n-1]-sol[j,(i)%N,n-1]))
            
    return sol

def beamwarm_acous(IC,dx,dt,T, lambd): # beam warming
    N = IC.shape[1]
    sol = np.zeros((2,N,int(T//dt)+1)) # rows: spatial, columns: temporal (+1 to account for t=0)
    sol[:,:,0] = decompose(IC)

    for j in range(lambd.shape[0]):
        if lambd[j] < 0:
            sol[j,:,:] = sol[j,::-1,:] 
        sol[j,:,:] = BeamWarming_lin(sol[j,:,0],dx,dt,T,abs(lambd[j]))
        if lambd[j] < 0:
            sol[j,:,:] = sol[j,::-1,:]       
    return sol

def beamwarm_lim_acous(IC,dx,dt,T, lambd): # beam warming
    N = IC.shape[1]
    sol = np.zeros((2,N,int(T//dt)+1)) # rows: spatial, columns: temporal (+1 to account for t=0)
    sol[:,:,0] = decompose(IC)

    for j in range(lambd.shape[0]):
        if lambd[j] < 0:
            sol[j,:,:] = sol[j,::-1,:] 
        sol[j,:,:] = BeamWarming_flux_limited_lin(sol[j,:,0],dx,dt,T,vanLeer,abs(lambd[j]))
        if lambd[j] < 0:
            sol[j,:,:] = sol[j,::-1,:]       
    return sol



IC2 =np.vstack(([f(x)*dx],np.ones_like(f(x))*dx))


fig,ax = plt.subplots(1,2)
sol = godunov_acous(IC2,dx,dt/np.max(eigval),T,eigval)
sol = sol/dx
final = recompose(sol[:,:,sol.shape[2]//2])
ax[0].plot(final[0,:])
ax[1].plot(final[1,:])
sol = beamwarm_acous(IC2,dx,dt/np.max(eigval),T,eigval)
sol = sol/dx
final = recompose(sol[:,:,sol.shape[2]//2])
ax[0].plot(final[0,:],".-")
ax[1].plot(final[1,:],".-")
sol = beamwarm_lim_acous(IC2,dx,dt/np.max(eigval),T,eigval)
sol = sol/dx
final = recompose(sol[:,:,sol.shape[2]//2])
ax[0].plot(final[0,:],".")
ax[1].plot(final[1,:],".")
# Exact
final = IC2/dx
ax[0].plot(final[0,:],"--")
ax[1].plot(final[1,:],"--")
plt.show()


# plt.figure()
# plt.title('p')
# plt.plot(sol_p[:,-1],label="Godunov")

# plt.figure()
# plt.title('u')
# plt.plot(sol_u[:,-1],label="Godunov")

# plt.show()