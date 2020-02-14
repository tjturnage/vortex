# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:55:03 2020
@author: thomas.turnage
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
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage.filters import gaussian_filter1d
from custom_cmaps import plts
cmap = plts['brown_gray_ramp']['cmap']

###############################################################################################

class VortexGrid2:
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


        #self.fig, self.ax = plt.subplots(1,1,figsize=(10,10),sharex=True)

        self.dimension=dimension
        self.skip_val = int(self.dimension/12)
        self.rotmax_fraction = rotmax_fraction
        #self.convergence = convergence
        self.convergence = -1
        self.translation = 1
        self.azshear_surge = []
        self.derivative = []
        self.x = np.linspace(-1,1,dimension)        
        self.y = np.linspace(-1,1,dimension)
        self.xx,self.yy = np.meshgrid(self.x,self.y)
        # set up x and y grids and mesh them into a 2d grid


    @property
    def distance(self):
        """
        2D grid based on dimension with each point containing value
        representing distance from origin (center of grid domain)
        """
        dist = np.sqrt(self.xx**2 + self.yy**2)
        return dist

    @property
    def rotmax_radius(self):
        """
        2D grid based on dimension with each point containing value
        representing distance from origin (center of grid domain)
        """
        rotmax_radius = self.rotmax_fraction * self.distance.max()
        return rotmax_radius

    @property
    def sin_angle(self):
        sin_ang = 2*np.pi*(self.yy/self.distance)
        return sin_ang     

    @property
    def cos_angle(self):
        cos_ang = 2*np.pi*(self.xx/self.distance)
        return cos_ang    

    @property
    def inner_radius_factor(self):
        return self.distance/self.rotmax_radius

    def u_rot_inner(self):
        u = (-1 * self.inner_radius_factor * self.sin_angle)
        return u

    def v_rot_inner(self):
        v = (self.inner_radius_factor * self.cos_angle)
        return v
    
    @property
    def outer_radius_factor(self):
        return 1/(self.distance/self.rotmax_radius)


    def masked_inner(self,grid):
        masked = ma.masked_array(grid,self.distance > self.rotmax_radius)        
        filled = masked.filled(fill_value=0)
        return filled

    def masked_outer(self,grid):
        masked = ma.masked_array(grid,self.distance <= self.rotmax_radius)        
        filled = masked.filled(fill_value=0)
        return filled

    def ma_inner(self,grid):
        ma_grid = self.masked_inner(grid)
        return ma_grid

    def ma_outer(self,grid):
        ma_grid = self.masked_outer(grid)
        return ma_grid

    
    #@property   
    def rotation_inner(self):
        u = (-1 * self.inner_radius_factor * self.sin_angle)
        v = (self.inner_radius_factor * self.cos_angle)
        u_fill = self.ma_inner(u)
        v_fill = self.ma_inner(v)
        return u_fill, v_fill

    #@property   
    def rotation_outer(self):
        u = (-1 * self.outer_radius_factor * self.sin_angle)
        v = (self.outer_radius_factor * self.cos_angle)
        u_fill = self.ma_outer(u)
        v_fill = self.ma_outer(v)
        return u_fill, v_fill


    @property   
    def convergence_inner(self):
        u = self.inner_radius_factor * self.convergence * self.cos_angle
        v = self.inner_radius_factor * self.convergence * self.sin_angle
        u_fill = self.ma_inner(u)
        v_fill = self.ma_inner(v)        
        return u_fill, v_fill  


    @property   
    def convergence_outer(self):
        u = self.outer_radius_factor * self.convergence * self.cos_angle
        v = self.outer_radius_factor * self.convergence * self.sin_angle
        u_fill = self.ma_outer(u)
        v_fill = self.ma_outer(v)        
        return u_fill, v_fill  

    @property   
    def convergence_U(self):
        conv_U = self.convergence_inner[0] + self.convergence_outer[0]
        return conv_U

    @property   
    def convergence_V(self):
        conv_V = self.convergence_inner[1] + self.convergence_outer[1]
        return conv_V

    @property   
    def convergence_speed(self):
        conv_mag = np.sqrt(self.convergence_U**2 + self.convergence_V**2)
        return conv_mag

    @property   
    def rotation_U(self):
        rot_U = self.rotation_inner()[0] + self.rotation_outer()[0]
        return rot_U

    @property    
    def rotation_V(self):
        rot_V = self.rotation_inner()[1] + self.rotation_outer()[1]
        return rot_V

    @property
    def rotation_convergence_U(self):
        return self.rotation_U + self.convergence_U

    @property
    def rotation_convergence_V(self):
        return self.rotation_V + self.convergence_V  

    @property
    def translation_factor(self):
        self.V_max = np.max(self.rotation_V)
        self.V_min = np.min(self.rotation_V)
        self.U_max = np.max(self.rotation_U)
        self.magnitude_max = np.max(np.sqrt(np.square(self.V_max) + np.square(self.U_max)))
        new_translation = self.translation * self.magnitude_max
        return new_translation

    @property
    def full_U(self):
        return self.rotation_convergence_U + self.translation_factor

    @property    
    def full_speed(self):
        full_mag = np.sqrt(self.full_U**2 + self.rotation_convergence_V**2)
        fully = full_mag/np.max(full_mag)
        full_list = fully.tolist()
        return full_list

    @property 
    def rot_V_trace(self):    
        if self.dimension%2 == 0:
            i = self.dimension/2
        else:
            i = self.dimension/2 + 1
        rot_V_list = self.rotation_V[int(i)].tolist()
        final = self.scale_trace(rot_V_list)
        return final

    @property 
    def conv_U_trace(self):    
        if self.dimension%2 == 0:
            i = self.dimension/2
        else:
            i = self.dimension/2 + 1
        self.conv_U_list = self.convergence_U[int(i)].tolist()
        final = self.scale_trace(self.conv_U_list)
        final2 = -1.0 * final
        return final2

    @property
    def azshear(self):
        self.derivative = []
        for i in range(0,len(self.rot_V_trace)):
            if i == 0:
                element = 0
            else:
                element = self.rot_V_trace[i] - self.rot_V_trace[i - 1]
            self.derivative.append(element)
        self.derivative[0] = self.derivative[-1]
        final = self.scale_trace(self.derivative)
        return final

    @property
    def divshear(self):
        self.derivative = []
        for i in range(0,len(self.conv_U_trace)):
            if i == 0:
                element = 0
            else:
                element = self.conv_U_trace[i] - self.conv_U_trace[i - 1]
            self.derivative.append(element)
        self.derivative[0] = self.derivative[-1]
        final = self.scale_trace(self.derivative)
        return final

    def rot_V_indices(self):    
        rot_min = np.min(self.rot_V_trace)
        rot_max = np.max(self.rot_V_trace)
        for t in range(0,len(self.rot_V_trace)):
            if self.rot_V_trace[t] == rot_max:
                max_index = t
            elif self.rot_V_trace[t] == rot_min:
                min_index = t
            else:
                pass
        return min_index,max_index


    def skip(self):
        skippy = int(self.skip_val)
        skip = (slice(None, None, skippy), slice(None, None, skippy))
        # plot quivers with substantial alpha
        return skip


    def quiver_full(self):
        #fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True)
        plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.full_U[self.skip()],self.rotation_convergence_V[self.skip()],color='k',alpha=0.5,zorder=10)
        plt.grid(False)

    def quiver_rotation(self):
        plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.rotation_U[self.skip()],self.rotation_V[self.skip()],color='k',alpha=0.5,zorder=10)


    def quiver_convergence(self):
        plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.convergence_U[self.skip()],self.convergence_V[self.skip()],color='k',alpha=0.5,zorder=10)
        plt.grid(False)

    def quiver_rotation_convergence(self):
        plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.rotation_convergence_U[self.skip()],self.rotation_convergence_V[self.skip()],color='k',alpha=0.5,zorder=10)


    def plot_collect(self,s):
        smoothed = gaussian_filter1d(s, sigma=1)
        points = np.array([self.x, smoothed]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plts['brown_gray_ramp']['cmap'],alpha=0.4,zorder=10)
        lc.set_array(self.y)
        lc.set_linewidth(3)
        return lc

    def plot_collections(self,s):
        p_cols = self.plot_collect(s)
        return p_cols

    def plot_collect_conv(self,s):
        smoothed = gaussian_filter1d(s, sigma=9)
        points = np.array([self.x, smoothed]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plts['brown_gray_ramp']['cmap'],alpha=0.4,zorder=10)
        lc.set_array(self.y)
        lc.set_linewidth(3)
        return lc

    def plot_collections_conv(self,s):
        p_cols = self.plot_collect_conv(s)
        return p_cols
    
    def azshear_plot(self):
        pl = self.plot_collections(self.azshear)
        pl.set_cmap(plts['brown_gray_ramp']['cmap'])
        pl.set_linewidth(6)
        pl.set_alpha(0.6)
        line = ax.add_collection(pl)
        return

    def divshear_plot(self):
        pl = self.plot_collections_conv(self.divshear)
        pl.set_cmap(plts['just_gray']['cmap'])
        pl.set_linewidth(6)
        pl.set_alpha(0.6)
        line = ax.add_collection(pl)
        return

    def rot_V_trace_plot(self):
        pl = self.plot_collections(self.rot_V_trace)
        pl.set_cmap(plts['brown_gray_ramp']['cmap'])
        pl.set_linewidth(6)
        pl.set_alpha(0.6)
        line = ax.add_collection(pl)
        return

    def conv_U_trace_plot(self):
        pl = self.plot_collections(self.conv_U_trace)
        pl.set_cmap(plts['just_gray']['cmap'])
        pl.set_linewidth(3)
        pl.set_alpha(0.6)
        plt.grid(True, lw=9, linestyle='-', color='w', alpha=0.3)
        line = ax.add_collection(pl)
        return


    def rot_V_contour(self):
        levels = np.arange(-8, 8, 1)
        cmap = plts['brown_ramp']['cmap']
        vmax = np.max(self.rotation_V)
        vmin = np.min(self.rotation_V)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        ax.contourf(self.xx,self.yy,self.rotation_V,levels,vmin=vmin,cmap=cmap, vmax=vmax,zorder=1, alpha=0.3)

    def convergence_speed_plot(self):
        levels = np.arange(0, 12, 0.25)
        cmap = plts['just_yellow']['cmap']
        vmax = np.max(self.convergence_speed)
        vmin = np.min(self.convergence_speed)
        levels = np.arange(0, int(vmax + 1), 0.25)
        #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        #ax.pcolormesh(self.xx,self.yy,self.convergence_speed,vmin=vmin,cmap=cmap, vmax=vmax,zorder=1, alpha=0.3)
        ax.contourf(self.xx,self.yy,self.convergence_speed,levels=levels,vmin=vmin,cmap=cmap, vmax=vmax,zorder=1, alpha=1)
        ax.contourf(self.xx,self.yy,self.convergence_speed,levels=levels,vmin=vmin,cmap=cmap, vmax=vmax,zorder=1, alpha=1)
    
    def norm(self):

        cmap = plts['brown_ramp']['cmap']
        levels = MaxNLocator(nbins=15).tick_values(0,12)
        normalize = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        return normalize

    def scale_trace_f(self,t):
            t_new =  t / (np.max(t) * 1.05)
            return t_new

    def scale_trace(self,t):
            ts = self.scale_trace_f(t)
            return ts

###############################################################################################

test = None
test = VortexGrid2()
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True)
plt.xlim(-1.0,1.0)
plt.ylim(-1.0,1.0)
plt.axis('scaled')
ax.set_yticks([0])
plt.grid(True)
ax.yaxis.set_ticklabels([])
ax.yaxis.set_ticks_position('none')
ax.set_xticks([10])
#ax.text(-0.14, -0.97, r'RADAR', fontsize=20,bbox=dict(facecolor='white', alpha=1),zorder=20) 

test.quiver_convergence()
test.convergence_speed_plot()
test.conv_U_trace_plot()
test.divshear_plot()
#test.rot_V_contour()
#test.rot_V_trace_plot()
#test.azshear_plot()
#azshear = test.arr_deriv(test.rot_V_trace())

image_dst_path = os.path.join(image_dir,'convshear3.png')
plt.savefig(image_dst_path,format='png',bbox_inches='tight')
