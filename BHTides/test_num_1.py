import numpy as np
from numpy import sin, cos, pi, sqrt
import pylab as plt
from mayavi import mlab

@mlab.animate(delay=10)
def anim():
    f = mlab.gcf()
    while 1:
        f.scene.camera.azimuth(-1)
        f.scene.render()
        yield

def test_plot3d():
    """Generates a pretty set of lines."""
    n_mer, n_long = 6, 11
    pi = numpy.pi
    dphi = pi / 1000.0
    phi = numpy.arange(0.0, 2 * pi + 0.5 * dphi, dphi)
    mu = phi * n_mer
    x = numpy.cos(mu) * (1 + numpy.cos(n_long * mu / n_mer) * 0.5)
    y = numpy.sin(mu) * (1 + numpy.cos(n_long * mu / n_mer) * 0.5)
    z = numpy.sin(n_long * mu / n_mer) * 0.5

    l = plot3d(x, y, z, numpy.sin(mu), tube_radius=0.025, colormap='Spectral')
    return l

thetax, phix = np.mgrid[0:np.pi:75j,0.*pi:2.*np.pi:150j]
pi = np.pi
phiz = np.linspace(0.,2*pi,num=150,endpoint=True)

q = 0.5
omegaH = q/4.   ####### Change This!!!!!!!
omega = sqrt(1/20.**3)

r_init = 1e-5

#r1 =  [(2 + 1*np.imag((-3*1j*(np.sin(th))**2*omega**2 - (9/2)*1j*np.sin(th)**2*omega**(8/3) - 16*(np.sin(th))**2*omega**3 
#         + 4.*(4.+5.*np.cos(th))*np.sin(th)**2 * omega**2*omegaH 
#                    )*np.exp(2*ph*1j))) for th,ph in zip(theta.ravel(),phi.ravel())]

#r1 = [(2 -  np.imag((-5/2*1j*np.sin(th)**3*omega**(8/3) - 35*sin(th)**3*omega**(11/3)+5*(7+3*np.cos(th))*sin(th)**3*omega**(8/3))*np.exp(3*ph*1j))) for th, ph in zip(theta.ravel(),phi.ravel())]
#r1 = [1.*(1. + 0.1*np.sin(2*ph))  for th,ph in zip(theta.ravel(),phi.ravel())]
r2 = [0.2 for th,ph in zip(thetax.ravel(),phix.ravel())]

#z3 = [-5 for th,ph in zip(theta.ravel(),phi.ravel())]

th = pi/2

r1z = [(2 + 1*np.imag((-3*1j*(np.sin(th))**2*omega**2 - (9/2)*1j*np.sin(th)**2*omega**(8/3) - 16*(np.sin(th))**2*omega**3 
        + 4.*(4.+5.*np.cos(th))*np.sin(th)**2 * omega**2*omegaH 
                   )*np.exp(2*ph*1j))) for ph in phiz]

#r1z = [(2 - np.imag((-5/2*1j*np.sin(th)**3*omega**(8/3) - 35*sin(th)**3*omega**(11/3)+5*(7+3*np.cos(th))*sin(th)**3*omega**(8/3))*np.exp(3*ph*1j))) for ph in phiz]

r1z = np.array(r1z)

x, y, z, X, Y, Z, costh, phi, r1 = np.loadtxt("/home/stepheno/programming/teukolsky/scott/tidalHSurf_1.log", skiprows=0, unpack=True)

costh = costh.reshape((101, 101))
phi = phi.reshape(costh.shape)

X = X.reshape(costh.shape)
Y = Y.reshape(costh.shape)
Z = Z.reshape(costh.shape)
x = x.reshape(costh.shape)
y = y.reshape(costh.shape)
z = z.reshape(costh.shape)

r1 = r1.reshape(costh.shape)







# data = []
# with open('tidalHSurf_1.log') as f:
#   for row in f.xreadlines():
#     row = row.strip().split(' ')
#     data.append(row)

# X, Y, Z, x, y, z, ... = zip(*data)


#x3, y3, r1z = X, Y, r1 if Z=0


print r1z.shape, phiz.shape

print r1.shape, x.shape, phi.shape 

x3 = r1z*cos(phiz)
y3 = r1z*sin(phiz)
z3 = [-5]*len(r1z)

#r1 =  [2 + np.imag(-3.*1j*np.sin(th)**2*np.exp(ph*1j)*omega**2) for th,ph in zip(theta.ravel(),phi.ravel())]
#r1 = np.array(r1).reshape(theta.shape)
r2 = np.array(r2).reshape(thetax.shape)

x1 = 1*np.sqrt(1-costh**2)*cos(phi)
y1 = 1*np.sqrt(1-costh**2)*sin(phi)
z1 = 1*costh

#x1 = r1*sin(theta)*cos(phi)
#y1 = r1*sin(theta)*sin(phi)
#z1 = r1*cos(theta)

x2 = r2*np.sin(thetax)*np.cos(phix)
y2 = r2*np.sin(thetax)*np.sin(phix)
z2 = r2*np.cos(thetax)




#z3 = [-5]*len(r1)
#mlab.figure()
#mlab.mesh(x1,y1,z1,scalars=r1)

#x2_ = -x2
#x2_+= 1

rot_angle = pi
Rz = np.array([[cos(rot_angle),-sin(rot_angle),0],
               [sin(rot_angle), cos(rot_angle),0],
               [0,             0,              1]])
B = np.dot(Rz,np.array([x2,y2,z2]).reshape((3,-1))) # we need to have a 3x3 times 3xN array
x2,y2,z2 = B.reshape((3,x2.shape[0],x2.shape[1])) # but we want our original shape back
x2 += -3 # simple translation


print X.shape, Y.shape, Z.shape, r1.shape

#mlab.options.offscreen = True

#mlab.figure()
#mlab.clf()

fig = mlab.gcf()

s = mlab.mesh(x,y,z,scalars=r1)
ms = s.mlab_source
# #mlab.mesh(x2,y2,z2,scalars=r2)
# #mlab.plot3d()

mesh2 = mlab.mesh(x2,y2,z2,scalars=r2)
mesh2_source = mesh2.mlab_source

#print x3.shape, y3.shape, z3.shape
#mlab.plot3d(x3,y3,z3,r1z)
#mlab.test_plot3d()
#mlab.show()

#a = anim()
omega = sqrt(1/(2.**3)) 
a = 0.99
omegaH =  a/(2*(1+sqrt(1-a**2))) 

print omega
print omegaH
# for i in range(1):
#     #xnew = x -0.1
#     xnew = x*np.cos(omegaH/20.) - y*np.sin(omegaH/20.)
#     #ynew = y + 0.05
#     ynew =  x*np.sin(omegaH/20.) + y*np.cos(omegaH/20.)
#     #z = z
#     #scalars = sqrt(xx*xx + yy*yy + zz*zz)
#     x = xnew
#     y = ynew

#     xnew = x2*np.cos(omega/20.) - y2*np.sin(omega/20.)
#     ynew =  x2*np.sin(omega/20.) + y2*np.cos(omega/20.)
#     x2 = xnew
#     y2 = ynew


#     ms.set(x=x,y=y) 
#     mesh2_source.set(x=x2,y=y2)
#     mlab.savefig('example'+str(i).zfill(3)+'.png')

xgrid = list()
ygrid = list()
zgrid = list()
sgrid = list()
connections = list()
N=100
index = 0

print "x shape:",x.shape
print "y shape:",y.shape
print "z shape:",z.shape

######################################################################

for i in range(19):
    xgrid.append(x[:,5*i])
    ygrid.append(y[:,5*i])
    zgrid.append(z[:,5*i])
    
    connections.append(np.vstack([np.arange(index, index + N - 1.5),np.arange(index + 1, index + N - 0.5)]).T)
    index += N

for i in range(20):
    xgrid.append(x[5*i,:])
    ygrid.append(y[5*i,:])
    zgrid.append(z[5*i,:])
    
    connections.append(np.vstack([np.arange(index, index + N - 1.5),np.arange(index + 1, index + N - 0.5)]).T)
    index += N

xgrid = np.hstack(xgrid)
ygrid = np.hstack(ygrid)
zgrid = np.hstack(zgrid)
connections = np.vstack(connections)

src = mlab.pipeline.scalar_scatter(xgrid, ygrid, zgrid)
src.mlab_source.dataset.lines = connections
lines = mlab.pipeline.stripper(src)
surf = mlab.pipeline.surface(lines,colormap='Accent',line_width=1,opacity=.4)
msurf = surf.mlab_source
###################################################################
# index = 0
# xgrid = list()
# ygrid = list()
# zgrid = list()
# sgrid = list()
# connections = list()
# for i in range(19):
#     xgrid.append(x[:,5*i])
#     ygrid.append(y[:,5*i])
#     zgrid.append(z[:,5*i])
    
#     connections.append(np.vstack([np.arange(index, index + N - 1.5),np.arange(index + 1, index + N - 0.5)]).T)
#     index += N

# for i in range(20):
#     xgrid.append(x[5*i,:])
#     ygrid.append(y[5*i,:])
#     zgrid.append(z[5*i,:])
    
#     connections.append(np.vstack([np.arange(index, index + N - 1.5),np.arange(index + 1, index + N - 0.5)]).T)
#     index += N

# xgrid = np.hstack(xgrid)
# ygrid = np.hstack(ygrid)
# zgrid = np.hstack(zgrid)
# connections = np.vstack(connections)

# src = mlab.pipeline.scalar_scatter(xgrid, ygrid, zgrid)
# src.mlab_source.dataset.lines = connections
# lines = mlab.pipeline.stripper(src)
# mlab.pipeline.surface(lines,colormap='Accent',line_width=1,opacity=.4)
####################################################################

for cnt in range(250):
    index = 0
    xgrid = list()
    ygrid = list()
    zgrid = list()
    sgrid = list()
    connections = list()
    for i in range(19):
        xgrid.append(x[:,5*i+cnt%5])
        ygrid.append(y[:,5*i+cnt%5])
        zgrid.append(z[:,5*i+cnt%5])
        
        connections.append(np.vstack([np.arange(index, index + N - 1.5),np.arange(index + 1, index + N - 0.5)]).T)
        index += N

    for i in range(20):
        xgrid.append(x[5*i,:])
        ygrid.append(y[5*i,:])
        zgrid.append(z[5*i,:])

        connections.append(np.vstack([np.arange(index, index + N - 1.5),np.arange(index + 1, index + N - 0.5)]).T)
        index += N

    xgrid = np.hstack(xgrid)
    ygrid = np.hstack(ygrid)
    zgrid = np.hstack(zgrid)
    connections = np.vstack(connections)

    #src = mlab.pipeline.scalar_scatter(xgrid, ygrid, zgrid)
    src.mlab_source.reset(x = xgrid,y = ygrid, z = zgrid)
    src.mlab_source.dataset.lines = connections
    lines = mlab.pipeline.stripper(src)
    #msurf.reset(lines=lines)
    surf = mlab.pipeline.surface(lines,colormap='Accent',line_width=1,opacity=.4)
#    fig.scene.reset_zoom()



    xnew = x*np.cos(2*pi*omega/(omegaH*N)) - y*np.sin(2*pi*omega/(N*omegaH))
    ynew =  x*np.sin(2*pi*omega/(N*omegaH)) + y*np.cos(2*pi*omega/(N*omegaH))
    x = xnew
    y = ynew

    xnew = x2*np.cos(2*pi*omega/(N*omegaH)) - y2*np.sin(2*pi*omega/(N*omegaH))
    ynew =  x2*np.sin(2*pi*omega/(N*omegaH)) + y2*np.cos(2*pi*omega/(N*omegaH))
    x2 = xnew
    y2 = ynew


    ms.set(x=x,y=y) 
    mesh2_source.set(x=x2,y=y2)
    mlab.savefig('example_num_'+str(cnt).zfill(3)+'.png')
#######################################################################
 
# data = mlab.pipeline.array2d_source(r1)

# # Use a greedy_terrain_decimation to created a decimated mesh
# terrain = mlab.pipeline.greedy_terrain_decimation(data)
# terrain.filter.error_measure = 'number_of_triangles'
# terrain.filter.number_of_triangles = 5000
# terrain.filter.compute_normals = True

# # Plot it black the lines of the mesh
# lines = mlab.pipeline.surface(terrain, color=(0, 0, 0),
#                                       representation='wireframe')
# # The terrain decimator has done the warping. We control the warping
# # scale via the actor's scale.
# lines.actor.actor.scale = [1, 1, 0.2]

# # Display the surface itself.
# surf = mlab.pipeline.surface(terrain, colormap='gist_earth',
#                                       vmin=1450, vmax=1650)
# surf.actor.actor.scale = [1, 1, 0.2]

# # Display the original regular grid. This time we have to use a
# # warp_scalar filter.
# warp = mlab.pipeline.warp_scalar(data, warp_scale=0.2)
# grid = mlab.pipeline.surface(warp, color=(1, 1, 1),
#                                       representation='wireframe')

# mlab.view(-17, 46, 143, [1.46, 8.46, 269.4])


mlab.show()

#for i in range (100):
#    mlab.camera.azimuth(i)
#plt.imshow(r1)
#plt.colorbar()

# mlab.figure()
# obj = mlab.mesh(x1,y1,z1,scalars=r1)
# P = mlab.pipeline
# scalar_cut_plane = P.scalar_cut_plane(obj,
#                                plane_orientation='y_axes',
#                                )

#scalar_cut_plane.enable_contours = True
#scalar_cut_plane.contour.filled_contours = True
#scalar_cut_plane.implicit_plane.widget.origin = np.array([  0.00000000e+00,   1.46059210e+00,  -2.02655792e-06])

#scalar_cut_plane.warp_scalar.filter.normal = np.array([ 0.,  1.,  0.])
#scalar_cut_plane.implicit_plane.widget.normal = np.array([ 0.,  1.,  0.])
#mlab.mesh(x2,y2,z2,scalars=r2)



# f = mlab.gcf()
# f.scene.show_axes = True
# f.scene.magnification = 4
# mlab.axes()

# dt = 0.01; N = 40
# ms = obj.mlab_source
# for k in xrange(N):
#     x1 = r1*sin(theta+k*dt)*cos(phi)
#     scalars = sin(x1**2 + y1**2)
#     ms.set(x1=x1, scalars=scalars)



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
