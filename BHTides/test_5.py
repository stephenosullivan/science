import numpy as np
from numpy import sin, cos, pi, sqrt
import pylab as plt
from mayavi import mlab
from scipy.optimize import newton

def roche(r,theta,phi,pot,q):
    lamr,nu = r*cos(phi)*sin(theta),cos(theta)
    return (pot - (1./r + q*(1./sqrt(1. - 2*lamr + r**2) - lamr) + 0.5*(q+1)*r**2*(1-nu**2)))

theta, phi = np.mgrid[0:np.pi:75j,0.0*pi:2.*np.pi:150j]

pot1,pot2 = 2.88,10.
q = 0.5
omega = sqrt(1/20.**3)

r_init = 1e-5

#r1 = [newton(roche,r_init,args=(th,ph,pot1,q)) for th,ph in zip(theta.ravel(),phi.ravel())]
#r2 = [newton(roche,r_init,args=(th,ph,pot2,1./q)) for th,ph in zip(theta.ravel(),phi.ravel())]
#r1 =  [(2 + 1*np.imag((-3*1j*(np.sin(th))**2*omega**2 - (9/2)*1j*np.sin(th)**2*omega**(8/3) - 16*(np.sin(th))**2*omega**3)*np.exp(ph*1j))) for th,ph in zip(theta.ravel(),phi.ravel())]
r1 = [1.*(1. + 0.1*np.sin(2*ph))  for th,ph in zip(theta.ravel(),phi.ravel())]
r2 = [0.2 for th,ph in zip(theta.ravel(),phi.ravel())]


#r1 =  [2 + np.imag(-3.*1j*np.sin(th)**2*np.exp(ph*1j)*omega**2) for th,ph in zip(theta.ravel(),phi.ravel())]
r1 = np.array(r1).reshape(theta.shape)
r2 = np.array(r2).reshape(theta.shape)

x1 = r1*sin(theta)*cos(phi)
y1 = r1*sin(theta)*sin(phi)
z1 = r1*cos(theta)

x2 = r2*np.sin(theta)*np.cos(phi)
y2 = r2*np.sin(theta)*np.sin(phi)
z2 = r2*np.cos(theta)

#mlab.figure()
#mlab.mesh(x1,y1,z1,scalars=r1)

x2_ = -x2
x2_+= 1

rot_angle = pi
Rz = np.array([[cos(rot_angle),-sin(rot_angle),0],
               [sin(rot_angle), cos(rot_angle),0],
               [0,             0,              1]])
B = np.dot(Rz,np.array([x2,y2,z2]).reshape((3,-1))) # we need to have a 3x3 times 3xN array
x2,y2,z2 = B.reshape((3,x2.shape[0],x2.shape[1])) # but we want our original shape back
x2 += 3 # simple translation


mlab.figure()
mlab.clf()
obj = mlab.mesh(x1,y1,z1,scalars=r1)

P = mlab.pipeline
scalar_cut_plane = P.scalar_cut_plane(obj,
                               plane_orientation='y_axes',
                               )

#scalar_cut_plane.enable_contours = True
#scalar_cut_plane.contour.filled_contours = True
#scalar_cut_plane.implicit_plane.widget.origin = np.array([  0.00000000e+00,   1.46059210e+00,  -2.02655792e-06])

#scalar_cut_plane.warp_scalar.filter.normal = np.array([ 0.,  1.,  0.])
#scalar_cut_plane.implicit_plane.widget.normal = np.array([ 0.,  1.,  0.])
#mlab.mesh(x2,y2,z2,scalars=r2)

f = mlab.gcf()
f.scene.show_axes = True
f.scene.magnification = 4
mlab.axes()

dt = 0.01; N = 40
ms = obj.mlab_source
for k in xrange(N):
    x1 = r1*sin(theta+k*dt)*cos(phi)
    scalars = sin(x1**2 + y1**2)
    ms.set(x1=x1, scalars=scalars)

#mlab.show()
# l = mlab.show()
# ms = l.mlab_source
# for i in range(10):
#     x1 = r1*sin(theta)*cos(phi+10*i)
#     scalars = np.sin(10*i)
#     ms.set(x=x,scalars=scalars)
#plt.imshow(r1)
#plt.colorbar()
#plt.figure()
#plt.imshow(r2)
#plt.colorbar()
