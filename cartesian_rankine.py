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
        """
        inputs:
               dimension : integer
                           number of horizontal grid points
         rotmax_fraction : float between 0 and 1
                           determines how far from the center of the plot the max rotation should bw
             convergence : float
                           value of convergence (negative) versus divergence (positive)
             translation : float
                           value of convergence (negative) versus divergence (positive)           
        """



        def __init__(self,dimension=200,rotmax_fraction=0.35,convergence = 0, translation = 0):

            
            rot_factor = {0.05: 5.868144057700324,
             0.1: 6.090232170097844,
             0.15: 6.165040427322349,
             0.2: 6.20253798984989,
             0.25: 6.225058809325617,
             0.3: 6.240080118927382,
             0.35: 6.2508126642787785,
             0.4: 6.258863500727778,
             0.45: 6.265126004120817,
             0.5: 6.27013642298489,
             0.55: 6.274236104646812,
             0.6: 6.277652661292794,
             0.65: 6.28054369503097,
             0.7: 6.283021792179115,
             0.75: 5.923843917544488,
             0.8: 5.553603672697958,
             0.85: 5.226921103715725,
             0.9: 4.93653659795374,
             0.95: 4.676718882271965,
             1.0: 4.442882938158366,
             1.05: 4.231317083960349,
             1.1: 4.0389844892348785,
             1.15: 3.8633764679637976,
             1.2: 3.702402448465306,
             1.25: 3.5543063505266934,
             1.3: 3.41760226012182,
             1.35: 3.2910243986358267,
             1.4: 3.173487812970262,
             1.45: 3.064057198729908,
             1.5: 2.961921958772244,
             1.55: 2.8663760891344303,
             1.6: 2.776801836348979,
             1.65: 2.692656326156586,
             1.7: 2.613460551857863,
             1.75: 2.5387902503762096,
             1.8: 2.46826829897687,
             1.85: 2.4015583449504683,
             1.9: 2.3383594411359825,
             1.95: 2.2784015067478802,
             2.0: 2.221441469079183}
            
            self.dimension=dimension
            self.rotmax_fraction = rotmax_fraction
            #self.convergence = convergence * rot_factor[self.rotmax_fraction]
            self.translation = translation * rot_factor[self.rotmax_fraction]
            self.convergence = convergence
            #self.translation = translation
            self.azshear = []


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
            self.V_max = self.V.max()
            self.magnitude = np.sqrt(self.U**2 + self.V**2)/4 * np.sqrt(2)
            self.magnitude_max = np.max(self.magnitude)

            self.U_ms = (self.U * units.meter / units.second)
            self.V_ms = (self.V * units.meter / units.second)
            self.vorticity = mpcalc.vorticity(self.U_ms, self.V_ms, 0.1 * units.meter, 0.1 * units.meter)
            self.vorticity_U = [self.vorticity[0],self.vorticity[1]*0]
            self.vorticity_V = [self.vorticity[0]*0,self.vorticity[1]]
            # trace1 refers to plot of V wrt x
            self.rotv_trace = self.V[int(dimension/2)]
            self.rotv_trace_scaled = self.rotv_trace / (np.max(self.rotv_trace) * 1.05)            

        

            for r in range(0,len(self.rotv_trace)):
                if r == 0:
                    self.azshear_element = 0
                else:
                    self.azshear_element = self.rotv_trace[r] - self.rotv_trace[r-1]
                
                self.azshear.append(self.azshear_element)

            self.azshear_scaled = self.azshear / (np.max(self.azshear) * 1.05)
            self.azshear_smoothed = gaussian_filter1d(self.azshear, sigma=1.5)



test = VortexGrid(200,0.30,0,0)


test2 = VortexGrid()
skip = (slice(None, None, 10), slice(None, None, 10))
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True)



levels = MaxNLocator(nbins=15).tick_values(0,12)

cmap = plts['brown_ramp']['cmap']

norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
vmax = test.V_max
vmin = -vmax


# color inbound/outbound
#continuous color field
cs = ax.pcolormesh(test.xx,test.yy,test.V,vmin=vmin,cmap=cmap, vmax=vmax,zorder=1)
# discrete filled contours
ax.contourf(test.xx,test.yy,test.V,vmin=vmin,cmap=cmap, vmax=vmax,zorder=1)


# quivers and density
# density with which to plot quivers. Greater numbers means more spacing between quivers
skip_val = 20
skip = (slice(None, None, skip_val), slice(None, None, skip_val))

# plot quivers with substantial alpha
plt.quiver(test.xx[skip],test.yy[skip],test.U[skip],test.V[skip],color='k',alpha=0.05,zorder=10)



x = test.x
y = test.rotv_trace_scaled      # rotational velocity trace
s = test.azshear_scaled         # azimuthal shear (NROT magnitude)


norm = plt.Normalize(y.min(), y.max())


# rotv plot
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=plts['brown_gray_ramp']['cmap'],norm=norm,zorder=10)
lc.set_array(y)
lc.set_linewidth(4)
line = ax.add_collection(lc)



# azshear plot
smoothed = gaussian_filter1d(s, sigma=1.8)
points2 = np.array([x, smoothed]).T.reshape(-1, 1, 2)
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
lc_az = LineCollection(segments2, cmap=plts['brown_gray_ramp']['cmap'],norm=norm,zorder=10)
lc_az.set_array(y)
lc_az.set_linewidth(4)
line = ax.add_collection(lc_az)


# plot characteristics
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1.0, 1.0)
plt.axis('scaled')
ax.set_yticks([10])
ax.set_xticks([10])
ax.grid(False)
fig.colorbar(cs,shrink=0.85)

plt.show()
            