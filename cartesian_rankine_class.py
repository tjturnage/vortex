# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 08:08:42 2020

@author: tjtur
"""


# -*- coding: utf-8 -*-
"""
Makes a Rankine Vortex in Cartesian coordinates
Rotational velocity (rotv) increases linearly from the center
Reaches a maximum value (magnitude_max) out to adefined radius (r)
Beyond that, rotv decreases as a function of (magnitude_max / r)
"""

import sys
import os

try:
    os.listdir('/usr')
    windows = False
    sys.path.append('/data/scripts/resources')
except:
    windows = True
    sys.path.append('C:/data/scripts/resources')

from reference_data import set_paths

data_dir,image_dir,archive_dir,gis_dir,py_call,placefile_dir = set_paths()


import numpy as np
import numpy.ma as ma
#import metpy.calc as mpcalc
#from metpy.units import units
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from custom_cmaps import plts
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.collections import LineCollection
#from my_functions import plot_settings
plt.rc('ytick', labelsize=14)

class VortexGrid:
        """
        inputs:
               dimension : integer
                           number of horizontal grid points
         rotmax_fraction : float between 0 and 1
                           determines how far from the center of the plot the max rotation should bw
             convergence : float
                           value of convergence (negative) versus divergence (positive)
             translation : float
                          fractional value related to maximum wind magnitude           
        """


        def __init__(self,dimension=200,rotmax_fraction=0.35,convergence = 0, translation = 0):


            self.dimension=dimension
            self.rotmax_fraction = rotmax_fraction
            self.convergence = convergence
            self.azshear = []
            self.azshear_surge = []
            
            # set up x and y grids and mesh them into a 2d grid
            self.x = np.linspace(-1,1,dimension)        
            self.y = np.linspace(-1,1,dimension)
            self.xx,self.yy = np.meshgrid(self.x,self.y)
            
            # calculate a distance from origin for each grid point
        def distance(self,xx,yy):
            self.distance = np.sqrt(self.xx**2 + self.yy**2)

            # the far corner grid points represent the max distance from the
            # center. Normalize against this to establish the radius of
            # maximum rotation
            self.max_distance = self.distance.max()
            self.rotmax_radius = self.rotmax_fraction * self.distance.max()

        def cartesian_grid(self):
            # Here we initialize 2 sets of U and V grids
            # the "_inner" represents rotation for points <= rotmax radius
            # the "_outer" represents rotation for points > rotmax radius
            self.grid = np.ndarray([self.dimension,self.dimension])
            self.grid.fill(0)
            #return grid

            #self.UU = cartesian_grid()
            #def outer_cartesian_grid(self):    
            #    self.U_outer = np.ndarray([dimension,dimension])
            #    self.U_outer.fill(0)
            #    self.V_outer = np.ndarray([dimension,dimension])
            #    self.V_outer.fill(0)

            # self.inner = self.cartesian_grid()
            # # Here we initialize 2 sets of U and V grids
            # # the "_inner" represents rotation for points <= rotmax radius
            # # the "_outer" represents rotation for points > rotmax radius
            # self.U_inner = np.ndarray([dimension,dimension])
            # self.U_inner.fill(0)
            # self.V_inner = np.ndarray([dimension,dimension])
            # self.V_inner.fill(0)

            # self.U_outer = np.ndarray([dimension,dimension])
            # self.U_outer.fill(0)
            # self.V_outer = np.ndarray([dimension,dimension])
            # self.V_outer.fill(0)


            self.sin_angle = 2*np.pi*(self.yy/self.distance)
            self.cos_angle = 2*np.pi*(self.xx/self.distance)


            # inner_radius factor is linear increase of rotv wrt radius
            # i.e., solid body rotation and is maximized at 1 
            self.inner_radius_factor = self.distance/(self.rotmax_radius)
            self.rotation_u_inner = (-1 * self.inner_radius_factor * self.sin_angle)
            self.rotation_v_inner = (self.inner_radius_factor * self.cos_angle)


            # outer_radius starts at 1 and decreases as 1/distance from rotmax
            # i.e., solid body rotation and is maximized at 1             
            self.outer_radius_factor = 1/(self.distance/self.rotmax_radius)
            self.rotation_u_outer = (-1 * self.outer_radius_factor * self.sin_angle)
            self.rotation_v_outer = (self.outer_radius_factor * self.cos_angle)


            # here we calculate U/V based only on rotv for distance <= rotv
            self.U_inner = self.rotation_u_inner
            self.V_inner = self.rotation_v_inner

            # create masked array that fills with U/V values only where distance > rotmax_radius
            self.U_inner_prefill = ma.masked_array(self.U_inner,self.distance > self.rotmax_radius)
            self.V_inner_prefill = ma.masked_array(self.V_inner,self.distance > self.rotmax_radius)

            # just put zeroes in the remaining grid points
            self.U_inner_filled = self.U_inner_prefill.filled(fill_value=0)
            self.V_inner_filled = self.V_inner_prefill.filled(fill_value=0)


            # doing the exact same thing, except this time for grid points outside of rotmax_radius
            self.U_outer = self.rotation_u_outer
            self.V_outer = self.rotation_v_outer
            self.U_outer_prefill = ma.masked_array(self.U_outer,self.distance <= self.rotmax_radius)
            self.V_outer_prefill = ma.masked_array(self.V_outer,self.distance <= self.rotmax_radius)
            self.U_outer_filled = self.U_outer_prefill.filled(fill_value=0)
            self.V_outer_filled = self.V_outer_prefill.filled(fill_value=0)


            # Now we can add inner and outer together, remembering that _inner and _outer
            # masked arrays were zero where the masks were. this is functionally equivalent to
            # compositing the different fields on either side of rotmax_radius
            self.U = (self.U_inner_filled + self.U_outer_filled)
            self.V = (self.V_inner_filled + self.V_outer_filled)

            
            # Only now is it possible to measure the max/min values of U,V as well as
            # maximum speed, because there seems to be a non-linear relationship between
            # this and rotmax fraction.
            self.V_max = np.max(self.V)
            self.V_min = np.min(self.V)
            self.U_max = np.max(self.U)
            self.U_min = np.min(self.U)
            self.magnitude_max = np.max(np.sqrt(np.square(self.V_max) + np.square(self.U_max)))
            self.translation = translation * self.magnitude_max
            self.U = self.U + self.translation


            # trace1 refers to plot of V wrt x
            # This uses a slice from the center of the y axis
            # len(y) is dimension
            self.rotv_trace = self.V[int(self.dimension/2)]
            self.rotv_trace_min = np.min(self.rotv_trace)
            self.rotv_trace_max = np.max(self.rotv_trace)            
            for t in range(0,len(self.rotv_trace)):
                if self.rotv_trace[t] == self.rotv_trace_max:
                    self.max_V_index = t
                    print('max index ' + str(t))
                elif self.rotv_trace[t] == self.rotv_trace_min:
                    self.min_V_index = t
                    print('min index ' + str(t))
                else:
                    pass

            self.rotv_trace_scaled = self.rotv_trace / (np.max(self.rotv_trace) * 1.05) 
            
                
            self.rotv_surge = []
            self.surge_factor = 2
            for i in range(0,len(self.rotv_trace)):
                self.val = self.rotv_trace[i]
                if i < self.min_V_index:
                    self.val_new = (1 + (self.surge_factor - 1)*(i/self.min_V_index)) * self.val
                    self.rotv_surge.append(self.val_new)
                elif i >= self.min_V_index and i <= self.max_V_index:
                    self.val_factor = (i - self.min_V_index)/(self.max_V_index - self.min_V_index)
                    self.rise = self.rotv_trace_max - (self.surge_factor * self.rotv_trace_min)
                    self.val = (self.surge_factor * self.rotv_trace_min) + (self.val_factor * self.rise)
                    self.rotv_surge.append(self.val)
                    #self.rotv_surge.append(-1 * np.abs(np.square((self.val ** 30.5))))
                else:
                    self.rotv_surge.append(self.val)

            self.rotv_surge_scaled = self.rotv_surge / (np.max(self.rotv_surge) * 1.05)
        

            # azimuthal shear calculated here by seeing how rotv changes wrt x
            for r in range(0,len(self.rotv_trace)):
                if r == 0:
                    self.azshear_element = 0
                else:
                    self.azshear_element = self.rotv_trace[r] - self.rotv_trace[r-1]
                
                self.azshear.append(self.azshear_element)

            self.azshear_scaled = self.azshear / (np.max(self.azshear) * 1.05)
            self.azshear_smoothed = gaussian_filter1d(self.azshear, sigma=1.5)


            for rs in range(0,len(self.rotv_surge)):
                if rs == 0:
                    self.azshear_surge_element = 0
                else:
                    self.azshear_surge_element = self.rotv_surge[rs] - self.rotv_surge[rs-1]
                
                self.azshear_surge.append(self.azshear_surge_element)

            self.azshear_scaled = self.azshear / (np.max(self.azshear) * 1.05)
            self.azshear_smoothed = gaussian_filter1d(self.azshear, sigma=1)

            self.azshear_surge_scaled = self.azshear_surge / (np.max(self.azshear_surge) * 1.05)
            self.azshear_surge_smoothed = gaussian_filter1d(self.azshear_surge, sigma=1)


            # ---------------------------------------------------------------------------------
            
            # What follows duplicates much of what was done already, only now we're adding
            # convergence and translations
        
            
            # Here we initialize 2 sets of U and V grids
            # This is for U/V including both convergence and translation 
            self.U_full_inner = np.ndarray([dimension,dimension])
            self.U_full_inner.fill(0)
            self.V_full_inner = np.ndarray([dimension,dimension])
            self.V_full_inner.fill(0)

            self.U_full_outer = np.ndarray([dimension,dimension])
            self.U_full_outer.fill(0)
            self.V_full_outer = np.ndarray([dimension,dimension])
            self.V_full_outer.fill(0)

            # using the same model of convergence with respect to radius as with
            # rotv. It seems that no normalization factor relative to speed is needed
            self.convergence_u_inner = self.inner_radius_factor * self.convergence * self.cos_angle
            self.convergence_v_inner = self.inner_radius_factor * self.convergence * self.sin_angle
            self.convergence_u_outer = self.outer_radius_factor * self.convergence * self.cos_angle
            self.convergence_v_outer = self.outer_radius_factor * self.convergence * self.sin_angle

            # here we calculate U/V based only on rotv for distance <= rotv
            self.U_inner_full = self.rotation_u_inner + self.convergence_u_inner + self.translation
            self.V_inner_full = self.rotation_v_inner + self.convergence_v_inner
            self.U_outer_full = self.rotation_u_outer + self.convergence_u_outer + self.translation
            self.V_outer_full = self.rotation_v_outer + self.convergence_v_outer


            # create masked array that fills with U_full/V_full values only where distance > rotmax_radius
            self.U_inner_full_prefill = ma.masked_array(self.U_inner_full,self.distance > self.rotmax_radius)
            self.V_inner_full_prefill = ma.masked_array(self.V_inner_full,self.distance > self.rotmax_radius)
            self.U_inner_full_filled = self.U_inner_full_prefill.filled(fill_value=0)
            self.V_inner_full_filled = self.V_inner_full_prefill.filled(fill_value=0)

            self.U_outer_full_prefill = ma.masked_array(self.U_outer_full,self.distance <= self.rotmax_radius)
            self.V_outer_full_prefill = ma.masked_array(self.V_outer_full,self.distance <= self.rotmax_radius)
            self.U_outer_full_filled = self.U_outer_full_prefill.filled(fill_value=0)
            self.V_outer_full_filled = self.V_outer_full_prefill.filled(fill_value=0)

            # Now we can add inner and outer together, remembering that _inner and _outer
            # masked arrays were zero where the masks were. this is functionally equivalent to
            # compositing the different fields on either side of rotmax_radius
            self.U_full = (self.U_inner_full_filled + self.U_outer_full_filled)
            self.V_full = (self.V_inner_full_filled + self.V_outer_full_filled)


            # trace1 refers to plot of V wrt x
            # This uses a slice from the center of the y axis
            # len(y) is dimension
            self.rotv_trace_full = self.V[int(self.dimension/2)]
            self.rotv_trace_full_min = np.min(self.rotv_trace_full)
            self.rotv_trace_full_max = np.max(self.rotv_trace_full)            
            for tf in range(0,len(self.rotv_trace_full)):
                if self.rotv_trace_full[tf] == self.rotv_trace_full_max:
                    self.max_V_index = tf
                elif self.rotv_trace[tf] == self.rotv_trace_full_min:
                    self.min_V_index = tf
                else:
                    pass

            self.rotv_trace_full_scaled = self.rotv_trace_full / (np.max(self.rotv_trace_full) * 1.05) 


            # trace1 refers to plot of V wrt x
            # This uses a slice from the center of the y axis
            # len(y) is dimension
            self.conv_trace_full = self.V_full[:,int(self.dimension/2)]
            self.conv_trace_full_min = np.min(self.conv_trace_full)
            self.conv_trace_full_max = np.max(self.conv_trace_full)            
            for c in range(0,len(self.conv_trace_full)):
                if self.conv_trace_full[c] == self.conv_trace_full_max:
                    self.max_conv_index = c
                elif self.conv_trace_full[c] == self.conv_trace_full_min:
                    self.min_conv_index = c
                else:
                    pass

            # trace1 refers to plot of V wrt x
            # This uses a slice from the center of the y axis
            # len(y) is dimension
            self.conv_u_trace_full = self.U_full[:,int(self.dimension/4)]
            self.conv_u_trace_full_min = np.min(self.conv_u_trace_full)
            self.conv_u_trace_full_max = np.max(self.conv_u_trace_full)            
            for cu in range(0,len(self.conv_u_trace_full)):
                if self.conv_u_trace_full[cu] == self.conv_u_trace_full_max:
                    self.max_conv_u_index = cu
                elif self.conv_u_trace_full[cu] == self.conv_u_trace_full_min:
                    self.min_conv_u_index = cu
                else:
                    pass

            self.conv_trace_full_scaled = self.conv_trace_full / (np.max(self.conv_trace_full) * 1.05) 
            self.convshear = []
            # convergence shear calculated here by seeing how conv changes wrt y
            for rc in range(0,len(self.conv_trace_full)):
                if rc == 0:
                    self.convshear_element = 0
                else:
                    self.convshear_element = self.conv_trace_full[rc] - self.conv_trace_full[rc-1]
                
                self.convshear.append(self.convshear_element)

            self.convshear_scaled = self.convshear / (np.max(self.convshear) * 1.05)
            self.convshear_smoothed = gaussian_filter1d(self.convshear, sigma=4)

            self.conv_u_trace_full_scaled = self.conv_u_trace_full / (np.max(self.conv_u_trace_full) * 1.05) 
            self.convshear_u = []
            # convergence shear calculated here by seeing how conv changes wrt y
            for ru in range(0,len(self.conv_u_trace_full)):
                if ru == 0:
                    self.convshear_u_element = 0
                else:
                    self.convshear_u_element = self.conv_u_trace_full[ru] - self.conv_u_trace_full[ru-1]
                
                self.convshear_u.append(self.convshear_u_element)

            self.convshear_u_scaled = self.convshear_u / (np.max(self.convshear_u) * 1.05)
            self.convshear_u_smoothed = gaussian_filter1d(self.convshear_u, sigma=4)


#test = VortexGrid(200,0.30,0,0)
test = VortexGrid(100,0.30,-1,0)
quiver = False

azshear = True
azshear_surge = True

rotv = True
rotv_surge =True
rotv_full = False

contour = True
grid = True

conv_full = False

from mpl_toolkits.axes_grid1 import make_axes_locatable
#test2 = VortexGrid()
skip = (slice(None, None, 10), slice(None, None, 10))
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True)
#fig, ax = plt.subplots(figsize=(5.5, 5.5))
#divider = make_axes_locatable(ax)
#axTracex= divider.append_axes("top", 1.2, pad=0.1,sharex=ax)
#axTracex.set_ylim([-1,1])

#axTracey = divider.append_axes("right", 1.2,        pad=0.1, sharey=ax)
#axTracey.set_ylim([-1,1])

# quivers and density
# density with which to plot quivers. Greater numbers means more spacing between quivers
if quiver:
    skip_val = 14
    skip = (slice(None, None, skip_val), slice(None, None, skip_val))
    # plot quivers with substantial alpha
    plt.quiver(test.xx[skip],test.yy[skip],test.U_full[skip],test.V_full[skip],color='k',alpha=0.5,zorder=10)


levels = MaxNLocator(nbins=15).tick_values(0,12)

cmap = plts['brown_ramp']['cmap']

norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
vmax = test.V_max
vmin = -vmax


if contour:
    levels = np.arange(-8, 8, 1)
    #ax.contourf(test.xx,test.yy,test.V,levels,vmin=vmin,cmap=cmap, vmax=vmax,zorder=1, alpha=0.6)
    ax.contourf(test.xx,test.yy,test.V,levels,vmin=vmin,cmap=cmap, vmax=vmax,zorder=1, alpha=0.3)

    #ax.pcolormesh(test.xx,test.yy,test.V,vmin=vmin,cmap=cmap, vmax=vmax,zorder=1, alpha=0.5)

x = test.x
y = test.y
y = test.rotv_trace_scaled
ys = test.rotv_surge_scaled
yf = test.rotv_trace_full_scaled 
cf = test.conv_trace_full_scaled 



s = test.azshear_scaled
ss = test.azshear_surge_scaled   # azimuthal shear (NROT magnitude)


norm_rotv = plt.Normalize(y.min(), y.max())
norm_rotv_surge = plt.Normalize(ys.min(), ys.max())
norm_rotv_full = plt.Normalize(ys.min(), ys.max())
norm_conv_full = plt.Normalize(cf.min(), cf.max())

# rotv plot
if rotv:
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #lc = LineCollection(segments, cmap=plts['brown_gray_ramp']['cmap'],norm=norm,zorder=10)
    #lc = LineCollection(segments, cmap=plts['just_gray']['cmap'],alpha=0.4,norm=norm,zorder=10)
    lc = LineCollection(segments, cmap=plts['just_gray']['cmap'],alpha=0.4,norm=norm,zorder=10)
    lc.set_array(y)
    lc.set_linewidth(6) # 4 orignially
    line = ax.add_collection(lc)

if rotv_surge:
    levels = np.arange(-8, 8, 1)
    #ax.contourf(test.xx,test.yy,test.V,levels,vmin=vmin,cmap=cmap, vmax=vmax,zorder=1, alpha=0.6)
    ax.contourf(test.xx,test.yy,test.V,levels,vmin=vmin,cmap=cmap, vmax=vmax,zorder=1, alpha=0.3)
    points_surge = np.array([x, ys]).T.reshape(-1, 1, 2)
    segments_surge = np.concatenate([points_surge[:-1], points_surge[1:]], axis=1)
    lc_surge = LineCollection(segments_surge, cmap=plts['just_yellow']['cmap'],norm=norm_rotv_surge,zorder=10)
    lc_surge.set_array(y)
    lc_surge.set_linewidth(6)
    line = ax.add_collection(lc_surge)

if rotv_full:
    points_full = np.array([x, yf]).T.reshape(-1, 1, 2)
    segments_full = np.concatenate([points_full[:-1], points_full[1:]], axis=1)
    lc_surge = LineCollection(segments_full, cmap=plts['brown_gray_ramp']['cmap'],norm=norm_rotv_full,zorder=10)
    lc_surge.set_array(y)
    lc_surge.set_linewidth(4)
    line = ax.add_collection(lc_surge)


if conv_full:
    conv_full = np.array([x, cf]).T.reshape(-1, 1, 2)
    #conv_full = np.array([x, cf]).reshape(-1, 1, 2)
    segments_conv_full = np.concatenate([conv_full[:-1], conv_full[1:]], axis=1)
    seg_trans = np.transpose(segments_conv_full)
    conv_f = LineCollection(segments_conv_full, cmap=plts['brown_gray_ramp']['cmap'],norm=norm_conv_full,zorder=10)
    #conv_f = LineCollection(seg_trans, cmap=plts['brown_gray_ramp']['cmap'],norm=norm_conv_full,zorder=10)
    conv_f.set_array(x)
    conv_f.set_linewidth(4)
    line = ax.add_collection(conv_f)
    #axTracey.scatter(cf,y)

# azshear plot
if azshear:
    smoothed = gaussian_filter1d(s, sigma=5)
    points2 = np.array([x, smoothed]).T.reshape(-1, 1, 2)
    segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
    #lc_az = LineCollection(segments2, cmap=plts['just_gray']['cmap'],norm=norm,alpha=0.4,zorder=10)
    lc_az = LineCollection(segments2, cmap=plts['just_gray']['cmap'],norm=norm,alpha=0.4,zorder=10)
    lc_az.set_array(y)
    lc_az.set_linewidth(6) #3 original
    line = ax.add_collection(lc_az)

if azshear_surge:
    smoothed_surge = gaussian_filter1d(ss, sigma=5)
    points_surge = np.array([x, smoothed_surge]).T.reshape(-1, 1, 2)
    segments_surge = np.concatenate([points_surge[:-1], points_surge[1:]], axis=1)
    lc_surge = LineCollection(segments_surge, cmap=plts['just_yellow']['cmap'],norm=norm_rotv_surge,alpha=0.9,zorder=10)
    lc_surge.set_array(y)
    lc_surge.set_linewidth(6)
    line = ax.add_collection(lc_surge)


# plot characteristics
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1.0, 1.0)
plt.axis('scaled')


ax.set_yticks([0])
plt.grid(True)
ax.yaxis.set_ticklabels([])
ax.yaxis.set_ticks_position('none')
ax.set_xticks([10])
if grid:
    ax.grid(True,color='k', linestyle='-', linewidth=2, alpha=1)
else:
    ax.grid(True,color='k', linestyle='-', linewidth=2, alpha=0)

ax.text(-0.14, -0.97, r'RADAR', fontsize=20,bbox=dict(facecolor='white', alpha=1),zorder=20)
image_dst_path = os.path.join(image_dir,'azshear_trace.png')
plt.savefig(image_dst_path,format='png',bbox_inches='tight')
            