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

```python
for nn in range(0,Nmax):
    NN[0,nn] = nn
    PPn[nn,:,:,:] = sp.special.lpmv(nn,nn,np.cos(gamma))
    if nn == 0:
        Pn[:,:,:] = PPn[nn,:,:,:]
        dxPn = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
        d2xPn = np.zeros((Nmax,Ndivx,Ndivy,Ndivz))
    else:
        Pn[:,:,:] = PPn[1,:,:,:]
        Pn_1[:,:,:] = PnR[nn,:,:,:]
        dxPn = nn*(np.cos(gamma)*Pn -Pn_1)/np.sin(gamma)
        d2xPn = nn*(2*np.cos(gamma)*Pn_1 + ((nn-1)*np.cos(gamma)**2 -nn -1)*Pn)/(np.sin(gamma)**2) + (np.cos(gamma)/(np.sin(gamma)**2) + (np.cos(gamma)/(np.sin(gamma)**2))*(nn*np.cos(gamma)*Pn -nn*Pn_1))
    PnR = Pn
    
    # Refelction Coefficients    
    x1 = k1*Ro
    x2 = k2*Ro

    Jc1 = sp.special.jv(nn + 0.5,x1)
    Jcc1 = sp.special.jv(nn + 1.5,x1)
    Jc2 = sp.special.jv(nn + 0.5,x2)
    Jcc2 = sp.special.jv(nn + 1.5,x2)

    Yc1 = sp.special.jv(-nn-0.5,x1)
    Ycc1 = sp.special.jv(0.5-nn,x1)
    Yc2 = sp.special.jv(-nn-0.5,x2)
    Ycc2 = sp.special.jv(0.5-nn,x2)

    # Spherical bessel/neumann functions

    j1 = np.sqrt(np.pi/(2*x1))*Jc1
    j2 = np.sqrt(np.pi/(2*x2))*Jc2
    y1 = (-1)**(nn+1)*np.sqrt(np.pi/(2*x1))*Yc1
    y2 = (-1)**(nn+1)*np.sqrt(np.pi/(2*x1))*Yc2

    # Derivatives of bessel fucntions

    dj1 = np.sqrt(np.pi/2)*(x1)**(-3/2)*(nn*Jc1 - np.dot(x1,Jcc1))
    dj2 = np.sqrt(np.pi/2)*(x2)**(-3/2)*(nn*Jc2 - np.dot(x2,Jcc2))
    dy1 = -(-1)**(nn+1)*np.sqrt(np.pi/2)*(x1**(-3/2)*((nn+1)*Yc1 + np.dot(x1,Ycc1)))
    dy2 = -(-1)**(nn+1)*np.sqrt(np.pi/2)*(x2**(-3/2)*((nn+1)*Yc2 + np.dot(x1,Ycc2)))

    # derivatives of e.g. (x1*j1)

    dxj1 = j1 + x1*dj1
    dxj2 = j2 + x2*dj2

    #Spherical hankel function
    h1 = j1 + 1j*y1 
    h2 = j2 + 1j*y2

    #derivative of h1
    dh1 = dj1 + 1j*dy1
    dh2 = dj2 + 1j*dy2

    #derivative of (x1*h1)
    dxh1 = h1 + x1*dh1
    dxh2 = h2 + x2*dh2

    # Centrifugal refelction coefficients
    Rs_F = (dxj2*j1 - dxj1*j2)/(dxj2*h1 - j2*dxh1)
    Rp_F = (np.dot(eps2,j2)*dxj1 - np.dot(eps1,j1)*dxj2)/(eps2*j2*dxh1 - np.dot(eps1,dxj2)*h1)
    Rs_P = (dxh2*h1 - dxh1*h2)/(j1*dxh2 - dxj1 * h2)
    Rp_P = (np.dot(eps2,h2)*dxh1 - np.dot(eps1,h1)*dxh2)/(np.dot(eps2,dxj1)*h2 - np.dot(eps1,j1)*dxh2)

    # Transmission Coefficients

    Ts_P = (j2*dxh2 - dxj2*h2*h2)/(j1*dxh2 - dxj1*h2)
    Tp_P = np.sqrt(np.dot(eps1,eps2))*(dxj2*h2 - j2*dxh2)/(np.dot(eps2,dxj1)*h2 - np.dot(eps1,np.dot(j1,dxh2)))

    # Coefficients
    Bs = -Rs_F
    Bp = -Rp_F
    Ds = (1/Ts_P)*(1-Rs_F*Rs_P) 
    Dp = (1/Tp_P)*(1-Rp_F*Rp_P)

    # Spherical Green function
    x1a = np.dot(k1,Ra)
    x1b = np.dot(k1,Rb)
    x2a = np.dot(k2,Ra)
    x2b = np.dot(k2,Rb)

    Jc1a  = sp.special.jv(nn+0.5,x1a)
    Jcc1a = sp.special.jv(nn+1.5,x1a)
    Jc1b  = sp.special.jv(nn+0.5,x1b)
    Jcc1b = sp.special.jv(nn+1.5,x1b)
    Yc1a  = sp.special.jv(-nn-0.5,x1a)
    Ycc1a = sp.special.jv( 0.5-nn,x1a)
    Yc1b  = sp.special.jv(-nn-0.5,x1b)
    Ycc1b = sp.special.jv( 0.5-nn,x1b)
    Jc2a  = sp.special.jv(nn+0.5,x2a)
    Jcc2a = sp.special.jv(nn+1.5,x2a)
    Jc2b  = sp.special.jv(nn+0.5,x2b)
    Jcc2b = sp.special.jv(nn+1.5,x2b)
    Yc2a  = sp.special.jv(-nn-0.5,x2a)
    Ycc2a = sp.special.jv( 0.5-nn,x2a)
    Yc2b  = sp.special.jv(-nn-0.5,x2b)
    Ycc2b = sp.special.jv( 0.5-nn,x2b)
    
    # spherical bessel/neumann functions
    j1a = np.sqrt(np.pi/(2*x1a))*Jc1a
    j1b = np.sqrt(np.pi/(2*x1b))*Jc1b
    y1a = (-1)**(nn+1)*np.sqrt(np.pi/(2*x1a))*Yc1a
    y1b = (-1)**(nn+1)*np.sqrt(np.pi/(2*x1b))*Yc1b
    j2a = np.sqrt(np.pi/(2*x2a))*Jc2a
    j2b = np.sqrt(np.pi/(2*x2b))*Jc2b
    y2a = (-1)**(nn+1)*np.sqrt(np.pi/(2*x2a))*Yc2a
    y2b = (-1)**(nn+1)*np.sqrt(np.pi/(2*x2b))*Yc2b
    
    # derivative of bessel/neumann functions
    dj1a = np.sqrt(np.pi/2)*(x1a)**(-3/2)*( nn*Jc1a - np.dot(x1a,Jcc1a) )
    dj1b = np.sqrt(np.pi/2)*(x1b)**(-3/2)*( nn*Jc1b - x1b*Jcc1b )
    dy1a = -(-1)**(nn+1)*np.sqrt(np.pi/2)*(x1a)**(-3/2)*((nn+1)*Yc1a + np.dot(x1a,Ycc1a) )
    dy1b = -(-1)**(nn+1)*np.sqrt(np.pi/2)*(x1b)**(-3/2)*((nn+1)*Yc1b + x1b*Ycc1b )
    dj2a = np.sqrt(np.pi/2)*(x2a)**(-3/2)*(nn*Jc2a - np.dot(x2a,Jcc2a))
    dj2b = np.sqrt(np.pi/2)*(x2b)**(-3/2)*(nn*Jc2b - x2b*Jcc2b)
    dy2a = -(-1)**(nn+1)*np.sqrt(np.pi/2)*(x2a)**(-3/2)*((nn+1)*Yc2a + np.dot(x2a,Ycc2a) )
    dy2b = -(-1)**(nn+1)*np.sqrt(np.pi/2)*(x2b)**(-3/2)*((nn+1)*Yc2b + x2b*Ycc2b )
    
    h1a   = j1a + 1j*y1a   # spherical Hankel function
    dh1a  = dj1a + 1j*dy1a # derivative of h1
    dxh1a = h1a + x1a*dh1a # derivative of (x1*h1)
    h2a   = j2a + 1j*y2a   # spherical Hankel function
    dh2a  = dj2a + 1j*dy2a # derivative of h2
    dxh2a = h2a + x2a*dh2a # derivative of (x2*h2)
    dxj1a = j1a + x1a*dj1a
    dxj1b = j1b + x1b*dj1b
    dxj2b = j2b + x2b*dj2b 
    
    h1b   = j1b + 1j*y1b   # spherical Hankel function
    dh1b  = dj1b + 1j*dy1b # derivative of h1
    dxh1b = h1b + x1b*dh1b # derivative of (x1*h1)
    h2b   = j2b + 1j*y2b   # spherical Hankel function
    dh2b  = dj2b + 1j*dy2b # derivative of h1
    dxh2b = h2b + x2b*dh2b # derivative of (x1*h1)
    
    #Outside Green's Functions:

    OGF_rr[nn,:,:,:] = st*(nn*(nn+1)*(2*nn+1)*( h1a*h1b*Pn[nn,:,:,:]/x1a/x1b )*Bp )*np.heaviside((Rb-Ro),0.5)  
    OGF_rr[np.isnan(OGF_rr)==1] = 0

    # Gro
    OGF_ro[nn,:,:,:] = st*((2*nn+1)*( h1b*dxh1a/x1a/x1b*dxPn[nn,:,:,:]*cosX2 )*Bp )*np.heaviside((Rb-Ro),0.5)
    OGF_ro[np.isnan(OGF_ro)==1] = 0

    # Gor
    OGF_or[nn,:,:,:] = st*((2*nn+1)*( dxh1b*h1a/x1a/x1b*dxPn[nn,:,:,:]*cosX1)*Bp )*np.heaviside((Rb-Ro),0.5)
    OGF_or[np.isnan(OGF_or)==1] = 0
                    
    Pp1 = +d2xPn[nn,:,:,:]*cosX1*cosX2 - dxPn[nn,:,:,:]*sinX1*sinX2/np.sin(gamma)  #(Legendre compact form)
    Pp2 = -d2xPn[nn,:,:,:]*sinX1*sinX2 + dxPn[nn,:,:,:]*cosX1*cosX2/np.sin(gamma) 
    Pp3 = -d2xPn[nn,:,:,:]*cosX1*sinX2 - dxPn[nn,:,:,:]*sinX1*cosX2/np.sin(gamma)
    Pp4 = d2xPn[nn,:,:,:]*cosX2*sinX1 + dxPn[nn,:,:,:]*sinX2*cosX1/np.sin(gamma)

    Pp1[np.isnan(Pp1)==1] = 0
    Pp2[np.isnan(Pp2)==1] = 0
    Pp3[np.isnan(Pp3)==1] = 0
    Pp4[np.isnan(Pp4)==1] = 0
    #d2xPn[np.isnan(d2xPn)==1] = 0

    Pp1T = np.sum(Pp1)
    Pp2T = np.sum(Pp2)
    Pp3T = np.sum(Pp3)
    Pp4T = np.sum(Pp4)
    d2xPnT = np.sum(d2xPn)
        
    # Goo
    OGF_ooTM[nn,:,:,:] = st*(2*nn+1)*(dxh1a*dxh1b/x1a/x1b*Pp1)/(nn*(nn+1))*Bp*np.heaviside((Rb-Ro),0.5) 
    OGF_ooTE[nn,:,:,:] = st*(2*nn+1)*(h1a*h1b*Pp2)/(nn*(nn+1))*Bs*np.heaviside((Rb-Ro),0.5)
    OGF_ooTM[np.isnan(OGF_ooTM)==1] = 0
    OGF_ooTE[np.isnan(OGF_ooTE)==1] = 0

    # Gqq
    OGF_qqTM[nn,:,:,:] = st*((2*nn+1)*( dxh1a*dxh1b/x1a/x1b*Pp2/nn/(nn+1) )*Bp )*np.heaviside((Rb-Ro),0.5)
    OGF_qqTE[nn,:,:,:] = st*((2*nn+1)*( h1a*h1b*Pp1 /nn/(nn+1))*Bs )*np.heaviside((Rb-Ro),0.5)
    OGF_qqTM[np.isnan(OGF_qqTM)==1] = 0
    OGF_qqTE[np.isnan(OGF_qqTE)==1] = 0


    # Grq
    OGF_rq[nn,:,:,:]=  st*(2*nn+1)*(h1b*dxh1a*(-dxPn[nn,:,:,:]*sinX2)/x1a/x1b)*Bp*np.heaviside((Rb-Ro),0.5)
    OGF_rq[np.isnan(OGF_rq)==1] = 0      
    
    
     # Gqr
    OGF_qr[nn,:,:,:]    =  st*(2*nn+1)*(dxh1b*h1a*(-dxPn[nn,:,:,:]*sinX1)/x1a/x1b)*Bp*np.heaviside(Rb-Ro,0.5)
    OGF_qr[np.isnan(OGF_qr)==1] = 0

    # Goq
    OGF_oqTM[nn,:,:,:]  =   st*((2*nn+1)/((nn+1)*(nn+2)))*(dxh1b*dxh1a*Pp3/x1a/x1b)*Bp*np.heaviside(Rb-Ro,0.5)
    OGF_oqTE[nn,:,:,:]  =   st*-((2*nn+1)/((nn+1)*(nn+2)))*(h1b*h1a*Pp4)*Bs*np.heaviside(Rb-Ro,0.5)
    OGF_oqTM[np.isnan(OGF_oqTM)==1] = 0;
    OGF_oqTE[np.isnan(OGF_oqTE)==1] = 0;

    # Gqo
    OGF_qoTM[nn,:,:,:]  =  st*((2*nn+1)/((nn+1)*(nn+2)))*(dxh1b*dxh1a*Pp4/x1a/x1b)*Bp*np.heaviside(Rb-Ro,0.5)
    OGF_qoTE[nn,:,:,:]  =  st*-((2*nn+1)/((nn+1)*(nn+2)))*(h1b*h1a*Pp3)*Bs*np.heaviside(Rb-Ro,0.5)
    OGF_qoTM[np.isnan(OGF_qoTM)==1] = 0
    OGF_qoTE[np.isnan(OGF_qoTE)==1] = 0

        
    # Inside Green's functions
         
    # Grr
    IGF_rr[nn,:,:,:] = st*(nn*(nn+1)*(2*nn+1)*( h1a*j2b*Pn[nn,:,:,:]/x1a/x2b)*Dp)*(1-np.heaviside(Rb-Ro,0.5))
    IGF_rr[np.isnan(IGF_rr)==1] = 0

    # Gro
    IGF_ro[nn,:,:,:] = st*((2*nn+1)*( j2b*dxh1a/x1a/x2b*dxPn[nn,:,:,:]*cosX2 )*Dp )*(1-np.heaviside(Rb-Ro,0.5))
    IGF_ro[np.isnan(IGF_ro)==1] = 0

    # Gor
    IGF_or[nn,:,:,:] = st*((2*nn+1)*( dxj2b*h1a/x1a/x2b*dxPn[nn,:,:,:]*cosX1 )*Dp )*(1-np.heaviside(Rb-Ro,0.5))
    IGF_or[np.isnan(IGF_or)==1] = 0

        
    # Goo
    IGF_ooTM[nn,:,:,:]  = st*(2*nn+1)*( dxh1a*dxj2b/x1a/x2b*Pp1 )/(nn*(nn+1))*Dp*(1-np.heaviside(Rb-Ro,0.5))
    IGF_ooTE[nn,:,:,:]  = st*(2*nn+1)*( h1a*j2b*Pp2)/(nn*(nn+1))*Ds*(1-np.heaviside(Rb-Ro,0.5))
    IGF_ooTM[np.isnan(IGF_ooTM)==1] = 0;
    IGF_ooTE[np.isnan(IGF_ooTE)==1] = 0;

    # Gqq
    IGF_qqTM[nn,:,:,:]  = st*((2*nn+1)*( dxh1a*dxj2b/x1a/x2b*Pp2/nn/(nn+1) )*Dp )*(1-np.heaviside(Rb-Ro,0.5))
    IGF_qqTE[nn,:,:,:]  = st*((2*nn+1)*( h1a*j2b*Pp1/nn/(nn+1))*Ds )*(1-np.heaviside(Rb-Ro,0.5))
    IGF_qqTM[np.isnan(IGF_qqTM)==1] = 0
    IGF_qqTE[np.isnan(IGF_qqTE)==1] = 0


    # Grq
    IGF_rq[nn,:,:,:] =  st*(2*nn+1)*(j2b*dxh1a*(-dxPn[nn,:,:,:]*sinX2)/x1a/x2b)*Dp*(1-np.heaviside(Rb-Ro,0.5))
    IGF_rq[np.isnan(IGF_rq)==1] = 0

    # Gqr
    IGF_qr[nn,:,:,:] =  st*(2*nn+1)*(dxj2b*h1a*(-dxPn[nn,:,:,:]*sinX1)/x1a/x2b)*Dp*(1-np.heaviside(Rb-Ro,0.5))
    IGF_qr[np.isnan(IGF_qr)==1]= 0

    # Goq
    IGF_oqTM[nn,:,:,:] = st*((2*nn+1)/((nn+1)*(nn+2)))*(dxj2b*dxh1a*Pp3/x1a/x2b)*Dp*(1-np.heaviside(Rb-Ro,0.5))
    IGF_oqTE[nn,:,:,:] = st*-((2*nn+1)/((nn+1)*(nn+2)))*(j2b*h1a*Pp4)*Ds*(1-np.heaviside(Rb-Ro,0.5))
    IGF_oqTM[np.isnan(IGF_oqTM)==1] = 0
    IGF_oqTE[np.isnan(IGF_oqTE)==1] = 0

    # Gqo
    IGF_qoTM[nn,:,:,:] = st*((2*nn+1)/((nn+1)*(nn+2)))*(dxj2b*dxh1a*Pp4/x1a/x2b)*Dp*(1-np.heaviside(Rb-Ro,0.5))
    IGF_qoTE[nn,:,:,:] = st*-((2*nn+1)/((nn+1)*(nn+2)))*(j2b*h1a*Pp3)*Ds*(1-np.heaviside(Rb-Ro,0.5))
    IGF_qoTM[np.isnan(IGF_qoTM)==1] = 0
    IGF_qoTE[np.isnan(IGF_qoTE)==1] = 0

    # Homogeneous Green's function
        
    # Grr
    OGF0_rr[nn,:,:,:]  =  ot*((2*nn+1)*nn*(nn+1))*((h1b*j1a)*np.heaviside(Rb-Ra,0.5)+(j1b*h1a)*(1-np.heaviside(Rb-Ra,0.5)))*Pn[nn,:,:,:]/x1a/x1b*np.heaviside(Rb-Ro,0.5)
    OGF0_rr[np.isnan(OGF0_rr)==1] = 0

        
    # Gro
    OGF0_ro[nn,:,:,:]  =  ot*((2*nn+1)*((h1b*dxj1a)*np.heaviside(Rb-Ra,0.5)+(j1b*dxh1a)*(1-np.heaviside(Rb-Ra,0.5)))/x1a/x1b*dxPn[nn,:,:,:]*cosX2 )*np.heaviside(Rb-Ro,0.5)
    OGF0_ro[np.isnan(OGF0_ro)==1] = 0

    # Gor
    OGF0_or[nn,:,:,:] = ot*(2*nn+1)*( ((dxh1b*j1a)*np.heaviside(Rb-Ra,0.5) + (dxj1b*h1a)*(1-np.heaviside(Rb-Ra,0.5)))/x1a/x1b*dxPn[nn,:,:,:]*cosX1 )*(np.heaviside(Rb-Ro,0.5))
    OGF0_or[np.isnan(OGF0_or)==1] = 0

    # Goo
    OGF0_ooTM[nn,:,:,:]  = ot*(2*nn+1)*( ((dxj1a*dxh1b)*np.heaviside(Rb-Ra,0.5)+(dxh1a*dxj1b)*(1-np.heaviside(Rb-Ra,0.5)))/x1a/x1b*Pp1 )/(nn*(nn+1))*(np.heaviside(Rb-Ro,0.5)) 
    OGF0_ooTE[nn,:,:,:]  = ot*(2*nn+1)*( (j1a*h1b)*np.heaviside(Rb-Ra,0.5) + (h1b*j1a)*(1-np.heaviside(Rb-Ra,0.5)))*Pp2/(nn*(nn+1))*(np.heaviside(Rb-Ro,0.5)) 
    OGF0_ooTM[np.isnan(OGF0_ooTM)==1] = 0;
    OGF0_ooTE[np.isnan(OGF0_ooTE)==1] = 0;

        # Gqq
    OGF0_qqTM[nn,:,:,:]  = ot*((2*nn+1)*( ((dxj1a*dxh1b)*np.heaviside(Rb-Ra,0.5)+(dxh1a*dxj1b)*(1-np.heaviside(Rb-Ra,0.5)))/x1a/x1b*Pp2/nn/(nn+1)))*(np.heaviside(Rb-Ro,0.5))
    OGF0_qqTE[nn,:,:,:]  = ot*((2*nn+1)*( ((j1a*h1b)*np.heaviside(Rb-Ra,0.5)+(h1a*j1b)*(1-np.heaviside(Rb-Ra,0.5)))*Pp1/nn/(nn+1)))*(np.heaviside(Rb-Ro,0.5))
    OGF0_qqTM[np.isnan(OGF0_qqTM)==1] = 0;
    OGF0_qqTE[np.isnan(OGF0_qqTE)==1] = 0;

        # Grq
    OGF0_rq[nn,:,:,:]    =  ot*(2*nn+1)*(((h1b*dxj1a)*np.heaviside(Rb-Ra,0.5) + (j1b*dxh1a)*(1-np.heaviside(Rb-Ra,0.5)))*(-dxPn[nn,:,:,:]*sinX2)/x1a/x1b)*(np.heaviside(Rb-Ro,0.5))
    OGF0_rq[np.isnan(OGF0_rq)==1] = 0

        # Gqr
    OGF0_qr[nn,:,:,:]    =  ot*(2*nn+1)*(((dxh1b*j1a)*np.heaviside(Rb-Ra,0.5)+ (dxj1b*h1a)*(1-np.heaviside(Rb-Ra,0.5)))*(-dxPn[nn,:,:,:]*sinX1)/x1a/x1b)*(np.heaviside(Rb-Ro,0.5))
    OGF0_qr[np.isnan(OGF0_qr)==1] = 0

        # Goq
    OGF0_oqTM[nn,:,:,:]  =   ot*((2*nn+1)/((nn+1)*(nn+1)))*(((dxh1b*dxj1a)*np.heaviside(Rb-Ra,0.5)+ (dxj1b*dxh1a)*(1-np.heaviside(Rb-Ra,0.5)))*Pp3/x1a/x1b)*(np.heaviside(Rb-Ro,0.5))
    OGF0_oqTE[nn,:,:,:]  =   ot*-((2*nn+1)/((nn+1)*(nn+1)))*(((h1b*j1a)*np.heaviside(Rb-Ra,0.5) + (j1b*h1a)*(1-np.heaviside(Rb-Ra,0.5)))*Pp4)*(np.heaviside(Rb-Ro,0.5))
    OGF0_oqTM[np.isnan(OGF0_oqTM)==1] = 0
    OGF0_oqTE[np.isnan(OGF0_oqTE)==1] = 0

        # Gqo
    OGF0_qoTM[nn,:,:,:]  =   ot*((2*nn+1)/((nn+1)*(nn+1)))*((dxh1b*dxj1a)*np.heaviside(Rb-Ra,0.5)+(dxj1b*dxh1a)*(1-np.heaviside(Rb-Ra,0.5)))*Pp4/x1a/x1b*(np.heaviside(Rb-Ro,0.5))
    OGF0_qoTE[nn,:,:,:]  =   ot*((2*nn+1)/((nn+1)*(nn+1)))*(((h1b*j1a)*np.heaviside(Rb-Ra,0.5)+(j1b*h1a)*(1-np.heaviside(Rb-Ra,0.5)))*Pp3)*(np.heaviside(Rb-Ro,0.5))
    OGF0_qoTM[np.isnan(OGF0_qoTM)==1] = 0
    OGF0_qoTE[np.isnan(OGF0_qoTE)==1] = 0
        
        # Magnetic Field Green's function - Homogeneous
         
        # Grr
    OGF0m_rr[nn,:,:,:]  =  np.zeros((Ndivy,Ndivx,Ndivz));
    OGF0m_rr[np.isnan(OGF0m_rr)==1] = 0;

        
        # Gro
    OGF0m_ro[nn,:,:,:]  =  ot*(-(2*nn+1)*((h1b*j1a)*np.heaviside(Rb-Ra,0.5)+(j1b*h1a)*(1-np.heaviside(Rb-Ra,0.5)))/x1b*dxPn[nn,:,:,:]*sinX2 )*np.heaviside(Rb-Ro,0.5)
    OGF0m_ro[np.isnan(OGF0m_ro)==1] = 0

        # Gor
    OGF0m_or[nn,:,:,:] = ot*(2*nn+1)*(((h1b*j1a)*np.heaviside(Rb-Ra,0.5) + (j1b*h1a)*(1-np.heaviside(Rb-Ra,0.5)))/x1a*dxPn[nn,:,:,:]*sinX1)*(np.heaviside(Rb-Ro,0.5))
    OGF0m_or[np.isnan(OGF0m_or)==1] = 0

        # Goo
    OGF0m_ooTM[nn,:,:,:]  = ot*(2*nn+1)*( ((j1a*dxh1b)*np.heaviside(Rb-Ra,0.5)+(h1a*dxj1b)*(1-np.heaviside(Rb-Ra,0.5)))/x1b*Pp3 )/(nn*(nn+1))*(np.heaviside(Rb-Ro,0.5)) ;
    OGF0m_ooTE[nn,:,:,:]  = ot*(2*nn+1)*( (dxj1a*h1b)*np.heaviside(Rb-Ra,0.5) + (j1b*dxh1a)*(1-np.heaviside(Rb-Ra,0.5)))/x1a*Pp4/(nn*(nn+1))*(np.heaviside(Rb-Ro,0.5))
    OGF0m_ooTM[np.isnan(OGF0m_ooTM)==1] = 0
    OGF0m_ooTE[np.isnan(OGF0m_ooTE)==1] = 0

        # Gqq
    OGF0m_qqTM[nn,:,:,:]  = ot*(-(2*nn+1)*( ((j1a*dxh1b)*np.heaviside(Rb-Ra,0.5)+(h1a*dxj1b)*(1-np.heaviside(Rb-Ra,0.5)))/x1b*Pp4/nn/(nn+1)))*(np.heaviside(Rb-Ro,0.5))
    OGF0m_qqTE[nn,:,:,:]  = ot*(-(2*nn+1)*( ((dxj1a*h1b)*np.heaviside(Rb-Ra,0.5)+(dxh1a*j1b)*(1-np.heaviside(Rb-Ra,0.5))*Pp3/x1a/nn/(nn+1)))*(np.heaviside(Rb-Ro,0.5)))
    OGF0m_qqTM[np.isnan(OGF0m_qqTM)==1] = 0
    OGF0m_qqTE[np.isnan(OGF0m_qqTE)==1] = 0

        # Grq
    OGF0m_rq[nn,:,:,:] = ot*-(2*nn+1)*(((h1b*j1a)*np.heaviside(Rb-Ra,0.5) + (j1b*h1a)*(1-np.heaviside(Rb-Ra,0.5)))*(dxPn[nn,:,:,:]*cosX2)/x1b)*(np.heaviside(Rb-Ro,0.5))
    OGF0m_rq[np.isnan(OGF0m_rq)==1] = 0

        # Gqr
    OGF0m_qr[nn,:,:,:] = ot*-(2*nn+1)*(((h1b*j1a)*np.heaviside(Rb-Ra,0.5)+ (j1b*h1a)*(1-np.heaviside(Rb-Ra,0.5)))*(dxPn[nn,:,:,:]*cosX1)/x1a)*(np.heaviside(Rb-Ro,0.5))
    OGF0m_qr[np.isnan(OGF0m_qr)==1] = 0

        # Goq
    OGF0m_oqTM[nn,:,:,:]  =   ot*(-(2*nn+1)/((nn+1)*(nn+1)))*(((dxh1b*j1a)*np.heaviside(Rb-Ra,0.5)+ (dxj1b*h1a)*(1-np.heaviside(Rb-Ra,0.5)))*Pp1/x1b)*(np.heaviside(Rb-Ro,0.5))
    OGF0m_oqTE[nn,:,:,:]  =   ot*((2*nn+1)/((nn+1)*(nn+1)))*(((h1b*dxj1a)*np.heaviside(Rb-Ra,0.5) + (j1b*dxh1a)*(1-np.heaviside(Rb-Ra,0.5)))*Pp2/x1a)*(np.heaviside(Rb-Ro,0.5))
    OGF0m_oqTM[np.isnan(OGF0m_oqTM)==1] = 0
    OGF0m_oqTE[np.isnan(OGF0m_oqTE)==1] = 0

        # Gqo
    OGF0m_qoTM[nn,:,:,:]  =   ot*((2*nn+1)/((nn+1)*(nn+1)))*((dxh1b*j1a)*np.heaviside(Rb-Ra,0.5)+(dxj1b*h1a)*(1-np.heaviside(Rb-Ra,0.5)))*Pp2/x1b*(np.heaviside(Rb-Ro,0.5))
    OGF0m_qoTE[nn,:,:,:]  =   ot*-((2*nn+1)/((nn+1)*(nn+1)))*(((h1b*dxj1a)*np.heaviside(Rb-Ra,0.5)+(j1b*dxh1a)*(1-np.heaviside(Rb-Ra,0.5)))*Pp1/x1a)*(np.heaviside(Rb-Ro,0.5))
    OGF0m_qoTM[np.isnan(OGF0m_qoTM)==1] = 0
    OGF0m_qoTE[np.isnan(OGF0m_qoTE)==1] = 0
        
        # Magnetic Green's functions - Outside
        
        # Grr
    OGFm_rr[nn,:,:,:] = np.zeros((Ndivy,Ndivx,Ndivz)) 
    OGFm_rr[np.isnan(OGFm_rr)==1] = 0

        # Gro
    OGFm_ro[nn,:,:,:] = st*(-(2*nn+1)*( h1b*h1a/x1b*dxPn[nn,:,:,:]*sinX2 )*Bs )*np.heaviside(Rb-Ro,0.5)
    OGFm_ro[np.isnan(OGFm_ro)==1] = 0

    # Gor
    OGFm_or[nn,:,:,:] = st*((2*nn+1)*(h1b*h1a/x1a*dxPn[nn,:,:,:]*sinX1)*Bp )*np.heaviside(Rb-Ro,0.5)
    OGFm_or[np.isnan(OGFm_or)==1] = 0

        # Goo
    OGFm_ooTM[nn,:,:,:]  = st*(2*nn+1)*(dxh1a*h1b/x1a*Pp4)/(nn*(nn+1))*Bp*np.heaviside(Rb-Ro,0.5)
    OGFm_ooTE[nn,:,:,:]  = st*(2*nn+1)*(h1a*dxh1b*Pp3)/x1b/(nn*(nn+1))*Bs*np.heaviside(Rb-Ro,0.5)
    OGFm_ooTM[np.isnan(OGFm_ooTM)==1] = 0
    OGFm_ooTE[np.isnan(OGFm_ooTE)==1] = 0


        # Gqq
    OGFm_qqTM[nn,:,:,:]  = st*(-(2*nn+1)*( dxh1a*h1b/x1a*Pp3/nn/(nn+1))*Bp )*np.heaviside(Rb-Ro,0.5)
    OGFm_qqTE[nn,:,:,:]  = st*(-(2*nn+1)*( h1a*dxh1b*Pp4/x1b/nn/(nn+1))*Bs )*np.heaviside(Rb-Ro,0.5)
    OGFm_qqTM[np.isnan(OGFm_qqTM)==1] = 0
    OGFm_qqTE[np.isnan(OGFm_qqTE)==1] = 0

        # Grq
    OGFm_rq[nn,:,:,:] =  st*-(2*nn+1)*(h1b*h1a*(dxPn[nn,:,:,:]*cosX2)/x1b)*Bs*np.heaviside(Rb-Ro,0.5)
    OGFm_rq[np.isnan(OGFm_rq)==1] = 0

        # Gqr
    OGFm_qr[nn,:,:,:] = st*-(2*nn+1)*(h1b*h1a*(dxPn[nn,:,:,:]*cosX1)/x1a)*Bp*np.heaviside(Rb-Ro,0.5)
    OGFm_qr[np.isnan(OGFm_qr)==1] = 0

        # Goq
    OGFm_oqTM[nn,:,:,:]  =   st*((2*nn+1)/((nn+1)*(nn+1)))*(h1b*dxh1a*Pp2/x1a)*Bp*np.heaviside(Rb-Ro,0.5)
    OGFm_oqTE[nn,:,:,:]  =   st*-((2*nn+1)/((nn+1)*(nn+1)))*(dxh1b*h1a*Pp1/x1b)*Bs*np.heaviside(Rb-Ro,0.5)
    OGFm_oqTM[np.isnan(OGFm_oqTM)==1] = 0
    OGFm_oqTE[np.isnan(OGFm_oqTE)==1] = 0

        # Gqo
    OGFm_qoTM[nn,:,:,:] = st*(-(2*nn+1)/((nn+1)*(nn+1)))*(h1b*dxh1a*Pp1/x1a)*Bp*np.heaviside(Rb-Ro,0.5)
    OGFm_qoTE[nn,:,:,:] = st*((2*nn+1)/((nn+1)*(nn+1)))*(dxh1b*h1a*Pp2/x1b)*Bs*np.heaviside(Rb-Ro,0.5)
    OGFm_qoTM[np.isnan(OGFm_qoTM)==1] = 0
    OGFm_qoTE[np.isnan(OGFm_qoTE)==1] = 0

    # Magnetic Green's functions - Inside 
         
    # Grr
    IGFm_rr[nn,:,:,:] = np.zeros((Ndivy,Ndivx,Ndivz))  
    IGFm_rr[np.isnan(IGFm_rr)==1] = 0


    # Gro
    IGFm_ro[nn,:,:,:] = st*(-(2*nn+1)*( j2b*h1a/x2b*dxPn[nn,:,:,:]*sinX2 )*Ds )*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_ro[np.isnan(IGFm_ro)==1] = 0

    # Gor
    IGFm_or[nn,:,:,:] = st*((2*nn+1)*( j2b*h1a/x1a*dxPn[nn,:,:,:]*sinX1)*Dp )*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_or[np.isnan(IGFm_or)==1] = 0

        
    # Goo
    IGFm_ooTM[nn,:,:,:]  = st*(2*nn+1)*( dxh1a*j2b/x1a*Pp4 )/(nn*(nn+1))*Dp*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_ooTE[nn,:,:,:]  = st*(2*nn+1)*( h1a*dxj2b*Pp3/x2b)/(nn*(nn+1))*Ds*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_ooTM[np.isnan(IGFm_ooTM)==1] = 0
    IGFm_ooTE[np.isnan(IGFm_ooTE)==1] = 0

    # Gqq
    IGFm_qqTM[nn,:,:,:]  = st*(-(2*nn+1)*( dxh1a*j2b/x1a*Pp3/nn/(nn+1) )*Dp )*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_qqTE[nn,:,:,:] = st*(-(2*nn+1)*( h1a*dxj2b*Pp4/x2b/nn/(nn+1))*Ds )*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_qqTM[np.isnan(IGFm_qqTM)==1] = 0
    IGFm_qqTE[np.isnan(IGFm_qqTE)==1] = 0


    # Grq
    IGFm_rq[nn,:,:,:]    =  st*-(2*nn+1)*(j2b*h1a*(dxPn[nn,:,:,:]*cosX2)/x2b)*Ds*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_rq[np.isnan(IGFm_rq)==1] = 0

    # Gqr
    IGFm_qr[nn,:,:,:]    =  st*-(2*nn+1)*(j2b*h1a*(dxPn[nn,:,:,:]*cosX1)/x1a)*Dp*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_qr[np.isnan(IGFm_qr)==1] = 0

    # Goq
    IGFm_oqTM[nn,:,:,:]  =   st*((2*nn+1)/((nn+1)*(nn+1)))*(j2b*dxh1a*Pp2/x1a)*Dp*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_oqTE[nn,:,:,:]  =   st*-((2*nn+1)/((nn+1)*(nn+1)))*(dxj2b*h1a*Pp1/x2b)*Ds*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_oqTM[np.isnan(IGFm_oqTM)==1] = 0
    IGFm_oqTE[np.isnan(IGFm_oqTE)==1] = 0
    
    # Gqo
    IGFm_qoTM[nn,:,:,:]  =   st*(-(2*nn+1)/((nn+1)*(nn+1)))*(j2b*dxh1a*Pp1/x1a)*Dp*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_qoTE[nn,:,:,:]  =   st*((2*nn+1)/((nn+1)*(nn+1)))*(dxj2b*h1a*Pp2/x2b)*Ds*(1-np.heaviside(Rb-Ro,0.5))
    IGFm_qoTM[np.isnan(IGFm_qoTM)==1] = 0
    IGFm_qoTE[np.isnan(IGFm_qoTE)==1] = 0    

    #Order Correction
    C_rr_TE[0,nn] = 0 
    C_ro_TE[0,nn]= 0 
    C_or_TE[0,nn] = 0    
    C_oo_TE[0,nn] = np.sum((IGF_ooTE[nn,:,:,:] + OGF_ooTE[nn,:,:,:])) 
    C_qq_TE[0,nn] =  np.sum((IGF_qqTE[nn,:,:,:] + OGF_qqTE[nn,:,:,:])) 
    C_oq_TE[0,nn] =  np.sum((IGF_oqTE[nn,:,:,:] + OGF_oqTE[nn,:,:,:]))
    C_qo_TE[0,nn] =  np.sum((IGF_qoTE[nn,:,:,:] + OGF_qoTE[nn,:,:,:]))
    C_rq_TE[0,nn] = 0      
    C_qr_TE[0,nn] = 0   
        
    C_rr_TM[0,nn] = np.sum((IGF_rr[nn,:,:,:] + OGF_rr[nn,:,:,:]))
    C_ro_TM[0,nn] = np.sum((IGF_ro[nn,:,:,:] + OGF_ro[nn,:,:,:]))
    C_or_TM[0,nn] = np.sum((IGF_or[nn,:,:,:] + OGF_or[nn,:,:,:]))   
    C_oo_TM[0,nn] = np.sum((IGF_ooTM[nn,:,:,:] + OGF_ooTM[nn,:,:,:])) 
    C_qq_TM[0,nn] = np.sum((IGF_qqTM[nn,:,:,:] + OGF_qqTM[nn,:,:,:])) 
    C_oq_TM[0,nn] = np.sum((IGF_oqTM[nn,:,:,:] + OGF_oqTM[nn,:,:,:]))
    C_qo_TM[0,nn] = np.sum((IGF_qoTM[nn,:,:,:] + OGF_qoTM[nn,:,:,:]))
    C_rq_TM[0,nn] = np.sum((IGF_rq[nn,:,:,:] + OGF_rq[nn,:,:,:]))    
    C_qr_TM[0,nn] = np.sum((IGF_qr[nn,:,:,:] + OGF_qr[nn,:,:,:]))

```

```python
GFs_rq = np.zeros((Ndivx,Ndivy,Ndivz))
GFs_qr = np.zeros((Ndivx,Ndivy,Ndivz))
GFs_oq = np.zeros((Ndivx,Ndivy,Ndivz))
GFs_qo = np.zeros((Ndivx,Ndivy,Ndivz))
GF0_rr = np.zeros((Ndivx,Ndivy,Ndivz))
GF0_or = np.zeros((Ndivx,Ndivy,Ndivz))
GF0_ro = np.zeros((Ndivx,Ndivy,Ndivz))
GF0_oo = np.zeros((Ndivx,Ndivy,Ndivz))
GF0_qq =  np.zeros((Ndivx,Ndivy,Ndivz))
GF0_rq =  np.zeros((Ndivx,Ndivy,Ndivz))
GF0_qr =  np.zeros((Ndivx,Ndivy,Ndivz))
GF0_oq =  np.zeros((Ndivx,Ndivy,Ndivz))
GF0_qo =  np.zeros((Ndivx,Ndivy,Ndivz))


GFms_rr = np.zeros((Ndivx,Ndivy,Ndivz))
GFms_ro = np.zeros((Ndivx,Ndivy,Ndivz))
GFms_or = np.zeros((Ndivx,Ndivy,Ndivz))
GFms_oo = np.zeros((Ndivx,Ndivy,Ndivz))
GFms_qq = np.zeros((Ndivx,Ndivy,Ndivz))
GFms_rq = np.zeros((Ndivx,Ndivy,Ndivz))
GFms_qr = np.zeros((Ndivx,Ndivy,Ndivz))
GFms_oq = np.zeros((Ndivx,Ndivy,Ndivz))
GFms_qo = np.zeros((Ndivx,Ndivy,Ndivz))

GF0m_rr = np.zeros((Ndivx,Ndivy,Ndivz))
GF0m_ro = np.zeros((Ndivx,Ndivy,Ndivz))
GF0m_or = np.zeros((Ndivx,Ndivy,Ndivz))
GF0m_oo = np.zeros((Ndivx,Ndivy,Ndivz))
GF0m_qq = np.zeros((Ndivx,Ndivy,Ndivz))
GF0m_rq = np.zeros((Ndivx,Ndivy,Ndivz))
GF0m_qr = np.zeros((Ndivx,Ndivy,Ndivz))
GF0m_oq = np.zeros((Ndivx,Ndivy,Ndivz))
GF0m_qo = np.zeros((Ndivx,Ndivy,Ndivz))


# Total Electric Green Function
GFs_rr[:,:,:] = 1j*k1/(4*np.pi)*np.sum(IGF_rr + OGF_rr,0)
GFs_ro[:,:,:] = 1j*k1/(4*np.pi)*np.sum(IGF_ro + OGF_ro , 0)
GFs_or[:,:,:] = 1j*k1/(4*np.pi)*np.sum(IGF_or + OGF_or , 0)
GFs_oo[:,:,:] = 1j*k1/(4*np.pi)*np.sum(IGF_ooTM + IGF_ooTE + OGF_ooTM + OGF_ooTE , 0)
GFs_qq[:,:,:] = 1j*k1/(4*np.pi)*np.sum(IGF_qqTM + IGF_qqTE + OGF_qqTM + OGF_qqTE , 0)
GFs_rq[:,:,:] = 1j*k1/(4*np.pi)*np.sum(IGF_rq + OGF_rq, 0)
GFs_qr[:,:,:] = 1j*k1/(4*np.pi)*np.sum(IGF_qr + OGF_qr, 0)
GFs_oq[:,:,:] = 1j*k1/(4*np.pi)*np.sum(IGF_oqTM + IGF_oqTE + OGF_oqTM + OGF_oqTE , 0)
GFs_qo[:,:,:] = 1j*k1/(4*np.pi)*np.sum(IGF_qoTM + IGF_qoTE + OGF_qoTM + OGF_qoTE , 0)

GF0_rr[:,:,:] = 1j*k1/(4*np.pi)*np.sum(OGF0_rr, 0)
GF0_ro[:,:,:] = 1j*k1/(4*np.pi)*np.sum(OGF0_ro , 0)
GF0_or[:,:,:] = 1j*k1/(4*np.pi)*np.sum(OGF0_or , 0)
GF0_oo[:,:,:] = 1j*k1/(4*np.pi)*np.sum(OGF0_ooTM + OGF0_ooTE , 0)
GF0_qq[:,:,:] = 1j*k1/(4*np.pi)*np.sum(OGF0_qqTM + OGF0_qqTE , 0)
GF0_rq[:,:,:] = 1j*k1/(4*np.pi)*np.sum(OGF0_rq, 0)
GF0_qr[:,:,:] = 1j*k1/(4*np.pi)*np.sum(OGF0_qr, 0)
GF0_oq[:,:,:] = 1j*k1/(4*np.pi)*np.sum(OGF0_oqTM + OGF0_oqTE , 0)
GF0_qo[:,:,:] = 1j*k1/(4*np.pi)*np.sum(OGF0_qoTM + OGF0_qoTE , 0)

# Total Magnetic Green Function
GFms_rr[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(IGFm_rr + OGFm_rr, 0)
GFms_ro[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(IGFm_ro + OGFm_ro , 0)
GFms_or[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(IGFm_or + OGFm_or , 0)
GFms_oo[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(IGFm_ooTM + IGFm_ooTE + OGFm_ooTM + OGFm_ooTE , 0)
GFms_qq[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(IGFm_qqTM + IGFm_qqTE + OGFm_qqTM + OGFm_qqTE , 0)
GFms_rq[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(IGFm_rq + OGFm_rq, 0)
GFms_qr[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(IGFm_qr + OGFm_qr, 0)
GFms_oq[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(IGFm_oqTM + IGFm_oqTE + OGFm_oqTM + OGFm_oqTE , 0)
GFms_qo[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(IGFm_qoTM + IGFm_qoTE + OGFm_qoTM + OGFm_qoTE , 0)

GF0m_rr[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(OGF0m_rr, 0)
GF0m_ro[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(OGF0m_ro , 0)
GF0m_or[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(OGF0m_or , 0)
GF0m_oo[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(OGF0m_ooTM + OGF0m_ooTE , 0)
GF0m_qq[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(OGF0m_qqTM + OGF0m_qqTE , 0)
GF0m_rq[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(OGF0m_rq, 0)
GF0m_qr[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(OGF0m_qr, 0)
GF0m_oq[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(OGF0m_oqTM + OGF0m_oqTE , 0)
GF0m_qo[:,:,:] = 1j*k1**2/(4*np.pi)*np.sum(OGF0m_qoTM + OGF0m_qoTE , 0)
```

```python
# Total Green function
# Note that o=theta, q=phi

GF_rr = GF0_rr + GFs_rr
GF_ro = GF0_ro + GFs_ro
GF_or = GF0_or + GFs_or
GF_oo = GF0_oo + GFs_oo
GF_rq = GF0_rq + GFs_rq
GF_qr = GF0_qr + GFs_qr
GF_oq = GF0_oq + GFs_oq
GF_qo = GF0_qo + GFs_qo
GF_qq = GF0_qq + GFs_qq

# Total Magnetic Green function
# Note that o=theta, q=phi

GFm_rr = GF0m_rr + GFms_rr
GFm_ro = GF0m_ro + GFms_ro
GFm_or = GF0m_or + GFms_or
GFm_oo = GF0m_oo + GFms_oo
GFm_rq = GF0m_rq + GFms_rq
GFm_qr = GF0m_qr + GFms_qr
GFm_oq = GF0m_oq + GFms_oq
GFm_qo = GF0m_qo + GFms_qo
GFm_qq = GF0m_qq + GFms_qq
```

```python
# Electric Fields :
greenColorMap = [np.zeros((1,132)),np.linspace(0,1,124)]
redColorMap = [np.linspace(1,0,124),np.zeros((1,132))]
colorMap = [[redColorMap],[greenColorMap],[np.zeros((1,256))]]
P_r = np.zeros((1,101))

if (animated_fields == 1):
    Nt = 51
    Nrepeat = 5
    Wt = np.array([np.linspace(0,2*np.pi*Nrepeat,Nt*Nrepeat)])
    ang = np.arange(0,2*np.pi,0.01)
    xc = Ro*np.cos(ang)
    yc=Ro*np.sin(ang)
    for tt in range(0,Nt*Nrepeat):
        wt = Wt[0,nn]
        E_r = ( GF_rr*U_pr + GF_ro * U_po + GF_rq * U_pq )* p_0*np.exp(-1j*wt)*(1j*w*mu0)
        E_o = ( GF_or*U_pr + GF_oo * U_po + GF_oq * U_pq )* p_0*np.exp(-1j*wt)*(1j*w*mu0)
        E_q = ( GF_qq*U_pq + GF_qr * U_pr + GF_qo * U_po )* p_0*np.exp(-1j*wt)*(1j*w*mu0)
    
        H_r = ( GFm_rr*U_pr + GFm_ro * U_po + GFm_rq * U_pq )* p_0*np.exp(-1j*wt)
        H_o = ( GFm_or*U_pr + GFm_oo * U_po + GFm_oq * U_pq )* p_0*np.exp(-1j*wt)
        H_q = ( GFm_qq*U_pq + GFm_qr * U_pr + GFm_qo * U_po )* p_0*np.exp(-1j*wt)
        
        OE_r = E_r[:,:,int((Ndivz+1)/2)]
        OE_o = E_o[:,:,int((Ndivz+1)/2)]
        OE_q = E_q[:,:,int((Ndivz+1)/2)]
        OE_q[:,int((Ndivx+1)/2)] = (OE_q[:,int((Ndivx+3)/2)] + OE_q[:,int((Ndivx-1)/2)])/2
        OE_qN = OE_q/np.amax(np.amax(OE_o.real))
        
        OH_r=H_r[:,:,int((Ndivz+1)/2)]
        OH_o=H_o[:,:,int((Ndivz+1)/2)]
        OH_q=H_q[:,:,int((Ndivz+1)/2)]
        OH_o[:,int((Ndivx+1)/2)]= (OH_o[:,int((Ndivx+3)/2)] + OH_o[:,int((Ndivx-1)/2)])/2
        OH_oN = OH_o/np.amax(np.amax(OH_o.real))
else:
        E_r = ( GF_rr*U_pr + GF_ro * U_po + GF_rq * U_pq )* p_0*(w**2*mu0)
        E_o = ( GF_or*U_pr + GF_oo * U_po + GF_oq * U_pq )*p_0*(w**2*mu0)
        E_q = ( GF_qq*U_pq + GF_qr * U_pr + GF_qo * U_po )* p_0*(w**2*mu0)
    
        H_r = ( GFm_rr*U_pr + GFm_ro * U_po + GFm_rq * U_pq )* p_0*(-1j*w)
        H_o = ( GFm_or*U_pr + GFm_oo * U_po + GFm_oq * U_pq )* p_0*(-1j*w)
        H_q = ( GFm_qq*U_pq + GFm_qr * U_pr + GFm_qo * U_po )*p_0*(-1j*w)
        
        
C_r_TE = C_rr_TE*U_pr + C_ro_TE*U_po + C_rq_TE*U_pq 
C_o_TE = C_or_TE*U_pr + C_oo_TE*U_po + C_oq_TE*U_pq 
C_q_TE = C_qr_TE*U_pr + C_qo_TE*U_po + C_qq_TE*U_pq  


C_r_TM = C_rr_TM*U_pr + C_ro_TM*U_po + C_rq_TM*U_pq 
C_o_TM = C_or_TM*U_pr + C_oo_TM*U_po + C_oq_TM*U_pq 
C_q_TM = C_qr_TM*U_pr + C_qo_TM*U_po + C_qq_TM*U_pq  


C_r    = C_r_TE + C_r_TM
C_o    = C_o_TE + C_o_TM
C_q    = C_q_TE + C_q_TM

C_total = np.sqrt(np.absolute(C_r)**2 + np.absolute(C_o)**2 + np.absolute(C_q)**2)
Cm = np.amax(C_total)
C_r_TE   = C_r_TE/Cm
C_r_TE[np.isnan(C_r_TE)==1] =0
C_o_TE   = C_o_TE/Cm  
C_o_TE[np.isnan(C_o_TE)==1] =0
C_q_TE   = C_q_TE/Cm
C_q_TE[np.isnan(C_q_TE)==1] =0

C_total_TM = np.sqrt(np.absolute(C_r_TM)**2 + np.absolute(C_o_TM)**2 + np.absolute(C_q_TM)**2)
E_total = np.absolute(E_r)**2 + np.absolute(E_o)**2 + np.absolute(E_q)**2
H_total = np.sqrt(np.absolute(E_r)**2 + np.absolute(E_o)**2 + np.absolute(E_q)**2)
C_total =  np.absolute(C_r)**2 + np.absolute(C_o)**2 + np.absolute(C_q)**2

E_r[np.isnan(E_r)==1] =0
E_o[np.isnan(E_o)==1] =0
E_q[np.isnan(E_q)==1] =0
E_total[np.isnan(E_total)==1] =0
H_total[np.isnan(H_total)==1] =0

#Final Fields

OE_r = E_r[:,:,int((Ndivz+1)/2)]
OE_o = E_o[:,:,int((Ndivz+1)/2)]
OE_q = E_q[:,:,int((Ndivz+1)/2)]
OE_total = E_total[:,:,int((Ndivz+1)/2)]


OH_r=H_r[:,:,int((Ndivz+1)/2)]
OH_o=H_o[:,:,int((Ndivz+1)/2)]
OH_q=H_q[:,:,int((Ndivz+1)/2)]
OH_total = H_total[:,:,int((Ndivz+1)/2)]

OP_r = 0.5*np.real(np.conjugate(OE_o)*OH_q - np.conjugate(OE_q)*OH_o)
OP_o = 0.5*np.real(np.conjugate(OE_q)*OH_r - np.conjugate(OE_r)*OH_q)
OP_q = 0.5*np.real(np.conjugate(OE_r)*OH_o - np.conjugate(OE_o)*OH_r)

OP_total = np.sqrt(np.absolute(OP_r)**2 + np.absolute(OP_o)**2 + np.absolute(OP_q)**2)

#Spin
eps = eps0*(np.heaviside(Rb-Ro,0.5)*(eps2-eps1) + eps1)
epsR = eps[:,:,1]

#Spin due to electric field
SE_r = ((c**2*epsR/4/w)*(OE_o*np.conj(OE_q) - OE_q*np.conj(OE_o))).imag
SE_o = ((c**2*epsR/4/w)*(OE_o*np.conj(OE_r) - OE_q*np.conj(OE_q))).imag
SE_q = ((c**2*epsR/4/w)*(OE_o*np.conj(OE_o) - OE_q*np.conj(OE_r))).imag

#Spin due to magnetic field
SH_r = c**2*mu0/4/w * (OH_o*np.conj(OH_q) - OH_q*np.conjugate(OH_o)).imag
SH_o = c**2*mu0/4/w * (OH_q*np.conj(OH_r) - OH_r*np.conjugate(OH_q)).imag
SH_q = c**2*mu0/4/w * (OH_r*np.conj(OH_o) - OH_o*np.conjugate(OH_r)).imag

S_r = SE_r + SH_r
S_o = SE_o + SH_o
S_q = SE_q + SH_q

S_total = np.sqrt(np.absolute(S_r)**2 + np.absolute(S_o)**2 + np.absolute(S_q)**2)

#Q Factor
L = len(OP_r)
for ii in range(1,102):
    thetaO = ((ii-1)/100)*360
    rO = Ro
    xO = rO*np.cos(thetaO)
    yO = rO*np.sin(thetaO)
    cx = np.min(np.absolute(xO-x))
    [dummy,CX] = np.where(np.absolute(xO-x) == cx)
    CX = CX[0]
    cy = np.min(np.absolute(yO-y))
    [dummy,CY] = np.where(np.absolute(yO-y) == cy)
    CY = CY[0]
    P_r[0,ii-1] = OP_r[CY,CX]
    
XRR         = ((2*np.sum(np.sum(np.heaviside(Ro-Rb[:,:,2],0.5))))/(Ro*101))   # Correction for the integrals
P_loss      = np.sum(P_r)*XRR
eps         = eps0*(np.heaviside(Rb-Ro,0.5)*(eps2-eps1) + eps1)
epsR[:,:]   = eps[:,:,int((Ndivz+1)/2)]
W_E  =  0.5*epsR*np.sqrt(abs(OE_r)**2  +  abs(OE_o)**2  +  abs(OE_q)**2)
W_H  =  0.5*mu0 *np.sqrt(abs(OH_r)**2  +  abs(OH_o)**2  +  abs(OH_q)**2)
W_stored = np.sum(np.sum((W_E + W_H)*np.heaviside(Ro-Rb[:,:,2],0.5)))
Q        = w*W_stored/P_loss
```
