Whispering gallery modes are known for possessing orbital angular momentum, however the interplay of local spin density, orbital angular momentum, and the near-field interaction with quantum emitters is far less explored. Here, we study the spin-orbit interaction of a circularly polarized dipole with the whispering gallery modes (WGMs) of a spherical resonator. Using an exact dyadic Green’s function approach, we show that the near-field interaction between the photonic spin of a circularly polarized dipole and the local electromagnetic spin density of whispering gallery modes gives rise to unidirectional behaviour where modes with either positive or negative orbital angular momentum are excited. We show that this is a manifestation of spin-momentum locking with the whispering gallery modes of the spherical resonator. We also discuss requirements for possible experimental demonstrations using Zeeman transitions in cold atoms or quantum dots, and outline potential applications of these previously overlooked properties. Our work firmly establishes local spin density, momentum and decay as a universal right-handed electromagnetic triplet for near-field light-matter interaction.

### References
*Farhad Khosravi, Cristian L. Cortes, and Zubin Jacob, "Spin photonics in 3D whispering gallery mode resonators," Opt. Express 27, 15846-15855 (2019) ; https://doi.org/10.1364/OE.27.015846*

``` python
import numpy as np
from scipy import linalg,special
import cmath
import matplotlib.pyplot as plt
import math
import scipy as sp
import warnings
warnings.filterwarnings('ignore')
```

```python
electric_plot = 0
magnetic_plot = 0
c_plot = 1
animated_fields = 0
spin_plot = 1
poynting_plot = 0
```

```python
Ro = 1177                            #nanoparticle radius
lamb = (0.518229038403211)*Ro        #TM Mode   n_r = 1 , l = 16

k0 = 2*np.pi/lamb                  # free-space wavevector
mu0 = 4*np.pi*1e-7*1e-9
eps0 = 8.85*1e-12*1e-9
c = 1/np.sqrt(mu0*eps0)
h = 1.054e-34*1e18
```

```python
# METAL
w = 2*np.pi/(lamb*1e-9)*3e8
wP = 6.3e15
gam = 1/5e-15

eps1 = 1
eps2 = 3 + 0*1j

k1 = np.dot(np.sqrt(eps1),k0)
k2 = np.sqrt(eps2)*k0

d = 100               #distance from the nanosphere
U_pr = 1  
U_po = 0
U_pq = 1j             # direction of the dipole
p_0 = 1               #dipole moment value
Ndiv  = 101           #number of division points in 3d space e.g. 100x100x100 point 
Ndivx = 101         
Ndivy = 101         
Ndivz = 3
Nmax = 50             #number of Legendre polynomials
Rd = 0                #radius to put the solutions inside into zero

# a is the dipole and b is the observation point
Ra = Ro + d           #dipole
Rm = Ro + 0.5*Ro     #maximum radial distance of observation point
phiA = 0
thetaA = np.pi/2

st = 1 #include scattered solutions
ot = 0 #include homogeneous solutions

x = np.array([np.linspace(-Rm,Rm,Ndivx)])
y = np.array([np.linspace(-Rm,Rm,Ndivy)])
z = np.array([np.linspace(-Rm,Rm,Ndivz)])

Y,X,Z = np.meshgrid(y,x,z)
Rb = np.sqrt(X**2 + Y**2 + Z**2)

QQ = np.zeros((Ndivx,Ndivy,Ndivz))
QQ = (Z < 0).astype(int)

thetaB = np.arctan(np.sqrt(X**2 + Y**2)/Z) + QQ*np.pi;
where = np.where(thetaB == 0)
thetaB[where[0], where[1], where[2]] = 0.001
```

```python
Q = np.zeros((Ndivx,Ndivy,Ndivz))
Q = (X < 0).astype(int)

T = ((X >= 0)*(Y < 0)).astype(int)
phiB = np.arctan(Y/X) + (Q*np.pi) + (T*2*np.pi)
gamma = (np.arccos(np.dot(np.cos(thetaA),np.cos(thetaB)) + np.dot(np.sin(thetaA),np.sin(thetaB))*np.cos(phiB-phiA))).real

cosX1 = (np.sin(thetaB)*np.cos(thetaA) - np.cos(thetaB)*np.sin(thetaA)*np.cos(phiB-phiA))/np.sin(gamma)
cosX2 = (np.sin(thetaA)*np.cos(thetaB) - np.cos(thetaA)*np.sin(thetaB)*np.cos(phiB-phiA))/np.sin(gamma)

sinX1 = (np.sin(thetaA)*np.sin(phiB-phiA))/np.sin(gamma)
sinX2 = (np.sin(thetaB)*np.sin(phiB-phiA))/np.sin(gamma)
```

```python
GFs_rr = np.zeros((Ndivx,Ndivy,Ndivz))
GFs_ro = np.zeros((Ndivx,Ndivy,Ndivz))
GFs_or = np.zeros((Ndivx,Ndivy,Ndivz))
GFs_oo = np.zeros((Ndivx,Ndivy,Ndivz))
GFs_qq = np.zeros((Ndivx,Ndivy,Ndivz))
GFs0_rr = np.zeros((Ndivx,Ndivy,Ndivz))
GFs0_oo = np.zeros((Ndivx,Ndivy,Ndivz))
GFs0_qq = np.zeros((Ndivx,Ndivy,Ndivz))

# Pre-Allocating for variables.
PnR = np.ones((Nmax,Ndivx,Ndivy,Ndivz))
Pn = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
PPn = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))

#Outside Green Functions
OGF_rr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF_ro = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF_or = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))

OGF_ooTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF_ooTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF_qqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF_qqTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF_rq = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF_qr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF_oqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF_oqTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF_qoTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF_qoTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))

#Inside Green Functions
IGF_rr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_ro = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_or = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_ooTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_ooTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_qqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_qqTE =np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_rq = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_qr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_oqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_oqTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_qoTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGF_qoTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))

#Homogeneous Greens Function
OGF0_rr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_ro = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_or = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_ooTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_ooTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_rq = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_qr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_oqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_oqTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_qoTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_qoTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_qqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0_qqTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))

# Magnetic Greens Function Homogeneous
OGF0m_rr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_ro = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_or = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_ooTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_ooTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_qqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_qqTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_rq = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_qr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_oqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_oqTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_qoTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGF0m_qoTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))

#Magnetic Green Function Outside
OGFm_rr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_ro = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_or = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_ooTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_ooTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_qqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_qqTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_rq = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_qr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_oqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_oqTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_qoTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
OGFm_qoTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))

# Magnetic Green Function Inside 
IGFm_rr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_ro = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_or = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_ooTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_ooTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_qqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_qqTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_qr = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_rq = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_oqTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_oqTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_qoTM = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
IGFm_qoTE = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
NN = np.zeros((1,Nmax))

C_rr_TE = np.zeros((1,Nmax))
C_ro_TE = np.zeros((1,Nmax))
C_or_TE = np.zeros((1,Nmax))
C_oo_TE = np.zeros((1,Nmax))
C_qq_TE = np.zeros((1,Nmax))
C_oq_TE = np.zeros((1,Nmax))
C_qo_TE = np.zeros((1,Nmax))
C_rq_TE = np.zeros((1,Nmax))
C_qr_TE = np.zeros((1,Nmax))

C_rr_TM = np.zeros((1,Nmax))
C_ro_TM = np.zeros((1,Nmax))
C_or_TM = np.zeros((1,Nmax))
C_oo_TM = np.zeros((1,Nmax))
C_qq_TM = np.zeros((1,Nmax))
C_oq_TM = np.zeros((1,Nmax))
C_qo_TM = np.zeros((1,Nmax))
C_rq_TM = np.zeros((1,Nmax))
C_qr_TM = np.zeros((1,Nmax))

```
