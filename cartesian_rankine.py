# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import os

try:
    os.listdir('/var/www')
    windows = False
    sys.path.append('/data/scripts/resources')
    from case_data import this_case
    case_date = this_case['date']
    rda = this_case['rda']
    base_gis_dir = '/data/GIS'

    case_dir = os.path.join('/data/radar',case_date,rda)
    base_dst_dir = os.path.join('/var/www/html/radar/images',case_date,rda)
    mosaic_dir = os.path.join(base_dst_dir,'mosaic')
except:
    windows = True
    sys.path.append('C:/data/scripts/resources')
    from case_data import this_case
    base_gis_dir = 'C:/data/GIS'
    topDir = 'C:/data'
    case_date = this_case['date']
    rda = this_case['rda']
    case_dir = os.path.join(topDir,case_date,rda)
    base_dst_dir = os.path.join(topDir,'images',case_date,rda)
    mosaic_dir = os.path.join(base_dst_dir,'mosaic') 



import numpy as np
import numpy.ma as ma
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.ticker import MaxNLocator
from custom_cmaps import plts
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.collections import LineCollection

class VortexGrid:
        def __init__(self,dimension=64,rotmax_fraction=0.25):
            self.dimension=dimension
            self.rotmax_fraction = rotmax_fraction
            self.convergence = 0
            self.translation = 0
            self.shear = []


            self.x = np.linspace(-1,1,dimension)        
            self.y = np.linspace(-1,1,dimension)
            self.xx,self.yy = np.meshgrid(self.x,self.y)
            self.distance = np.sqrt(self.xx**2 + self.yy**2)
            self.max_distance = self.distance.max()
            self.rotmax_radius = self.rotmax_fraction * self.distance.max()

            self.U_inner = np.ndarray([dimension,dimension])
            self.U_inner.fill(0)
            self.V_inner = np.ndarray([dimension,dimension])
            self.V_inner.fill(0)

            self.U_outer = np.ndarray([dimension,dimension])
            self.U_outer.fill(0)
            self.V_outer = np.ndarray([dimension,dimension])
            self.V_outer.fill(0)
            self.vort_zero = np.ndarray([dimension,dimension])
            self.vort_zero.fill(0)



            #self.inner_radius_factor = self.distance
            #self.outer_radius_factor = 1/self.distance
            self.inner_radius_factor = self.distance/(self.rotmax_radius)
            self.outer_radius_factor = 1/(self.distance/self.rotmax_radius)
            self.sin_angle = 2*np.pi*(self.yy/self.distance)
            self.cos_angle = 2*np.pi*(self.xx/self.distance)

            self.rotation_u_inner = (-1 * self.inner_radius_factor * self.sin_angle)
            self.rotation_v_inner = (self.inner_radius_factor * self.cos_angle)
            self.rotation_u_outer = (-1 * self.outer_radius_factor * self.sin_angle)
            self.rotation_v_outer = (self.outer_radius_factor * self.cos_angle)
            
            self.convergence_u_inner = self.inner_radius_factor * self.convergence * self.cos_angle
            self.convergence_v_inner = self.inner_radius_factor * self.convergence * self.sin_angle
            self.convergence_u_outer = self.outer_radius_factor * self.convergence * self.cos_angle
            self.convergence_v_outer = self.outer_radius_factor * self.convergence * self.sin_angle

#
            self.U_inner = self.rotation_u_inner + self.convergence_u_inner + self.translation
            self.V_inner = self.rotation_v_inner + self.convergence_v_inner

            self.U_inner_prefill = ma.masked_array(self.U_inner,self.distance > self.rotmax_radius)
            self.V_inner_prefill = ma.masked_array(self.V_inner,self.distance > self.rotmax_radius)
            self.U_inner_filled = self.U_inner_prefill.filled(fill_value=0)
            self.V_inner_filled = self.V_inner_prefill.filled(fill_value=0)

            self.U_outer = self.rotation_u_outer + self.convergence_u_outer + self.translation
            self.V_outer = self.rotation_v_outer + self.convergence_v_outer

            self.U_outer_prefill = ma.masked_array(self.U_outer,self.distance <= self.rotmax_radius)
            self.V_outer_prefill = ma.masked_array(self.V_outer,self.distance <= self.rotmax_radius)
            self.U_outer_filled = self.U_outer_prefill.filled(fill_value=0)
            self.V_outer_filled = self.V_outer_prefill.filled(fill_value=0)


            self.U = (self.U_inner_filled + self.U_outer_filled)
            self.V = (self.V_inner_filled + self.V_outer_filled)
            self.magnitude = np.sqrt(self.U**2 + self.V**2)/4 * np.sqrt(2)

            self.U_ms = (self.U * units.meter / units.second)
            self.V_ms = (self.V * units.meter / units.second)
            self.vorticity = mpcalc.vorticity(self.U_ms, self.V_ms, 0.1 * units.meter, 0.1 * units.meter)
            self.vorticity_U = [self.vorticity[0],self.vorticity[1]*0]
            self.vorticity_V = [self.vorticity[0]*0,self.vorticity[1]]
            self.trace1 = self.V[int(dimension/2)]
            self.trace1_scaled = self.trace1 / (np.max(self.trace1) * 1.05)            

        

            for r in range(0,len(self.trace1)):
                if r == 0:
                    self.shear_element = 0
                else:
                    self.shear_element = self.trace1[r] - self.trace1[r-1]
                
                self.shear.append(self.shear_element)

            self.shear_scaled = self.shear / (np.max(self.shear) * 1.05)
            self.shear_smoothed = gaussian_filter1d(self.shear, sigma=1.5)
test = VortexGrid()
skip = (slice(None, None, 2), slice(None, None, 2))
fig, ax = plt.subplots(1,1,figsize=(8,8),sharex=True)



levels = MaxNLocator(nbins=15).tick_values(0,12)
cmap = plt.get_cmap('PiYG_r')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)


cs = ax.pcolormesh(test.xx,test.yy,test.V,vmin=-8,cmap=plts['GTB_light']['cmap'], vmax=8,zorder=1)

fig.colorbar(cs,shrink=0.85)

skip = (slice(None, None, 4), slice(None, None, 4))
#plt.quiver(test.xx[skip],test.yy[skip],test.U[skip],test.V[skip],color='k',alpha=0.2,zorder=10)
plt.axis('scaled')
ax.set_yticks([0])
ax.set_xticks([0])
ax.grid(True)

ax.plot(test.x,test.shear_scaled,':',linewidth=1,color='k',alpha=0.6)

x = test.x
y = test.trace1_scaled

s = test.shear_scaled

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm = plt.Normalize(y.min(), y.max())
lc = LineCollection(segments, cmap=plts['GTB']['cmap'], norm=norm,zorder=10)
lc.set_array(y)
lc.set_linewidth(2)
line = ax.add_collection(lc)

points = np.array([x, s]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)


norm = plt.Normalize(s.min(), s.max())
lc = LineCollection(segments, cmap=plts['GTB']['cmap'], norm=norm,zorder=10)
lc.set_array(s)
lc.set_linewidth(2)
line = ax.add_collection(lc)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1.0, 1.0)

plt.show()
            