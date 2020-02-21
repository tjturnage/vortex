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
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage.filters import gaussian_filter1d
from custom_cmaps import plts
green_white_pink = plts['green_white_pink']['cmap']
green_white_brown = plts['green_white_brown']['cmap']
green_gray_pink = plts['green_gray_pink']['cmap']
blue_black_red = plts['blue_black_red']['cmap']
red_black_blue = plts['red_black_blue']['cmap']
blue_gray_red = plts['blue_gray_red']['cmap']
divy = plts['DivShear_Storm']['cmap']
white_brown = plts['white_brown']['cmap']
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
        self.skip_val_1 = int(self.dimension/12)
        self.skip_val_2 = int(self.dimension/6)
        self.rotmax_fraction = rotmax_fraction
        #self.convergence = convergence
        self.convergence = -1
        self.translation = 0
        self.azshear_surge = []
        self.derivative = []
        self.line_convergence = []
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

    @property
    def u_rot_inner(self):
        u = (-1 * self.inner_radius_factor * self.sin_angle)
        return u

    @property
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

    def skip(self):
        skippy = int(self.skip_val_1)
        skip = (slice(None, None, skippy), slice(None, None, skippy))
        # plot quivers with substantial alpha
        return skip

    def skip2(self):
        skippy = int(self.skip_val_2)
        skip = (slice(None, None, skippy), slice(None, None, skippy))
        # plot quivers with substantial alpha
        return skip
    
    ##################################################################
    ###########################  Convergence #########################
    ##################################################################

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

    def convergence_V_contour_plot(self):
        levels = np.arange(-2.01, 2.01, 0.2)
        vmax = np.max(self.convergence_V)
        vmin = np.min(self.convergence_V)
        ax.contourf(self.xx,self.yy,self.convergence_V,levels=levels,vmin=vmin,cmap=red_black_blue, vmax=vmax,zorder=1, alpha=1)
        ax.contourf(self.xx,self.yy,self.convergence_V,levels=levels,vmin=vmin,cmap=red_black_blue, vmax=vmax,zorder=1, alpha=1)
        return

    @property   
    def convergence_speed(self):
        conv_mag = np.sqrt(self.convergence_U**2 + self.convergence_V**2)
        return conv_mag/(np.max(conv_mag) * 1.05)

    @property 
    def convergence_U_trace(self):    
        if self.dimension%2 == 0:
            i = self.dimension/2
        else:
            i = self.dimension/2 + 1
        self.conv_U_list = self.convergence_U_speed[int(i)].tolist()
        final = self.conv_U_list/(np.max(self.conv_U_list))
        return final

    def convergence_U_trace_plot(self):
        pl = self.plot_collections(self.convergence_U_trace,6,green_white_brown)
        pl.set_linewidth(3)
        pl.set_alpha(0.6)
        plt.grid(True, lw=9, linestyle='-', color='w', alpha=0.3)
        line = ax.add_collection(pl)
        return

    @property 
    def convergence_V_trace_plot(self):
        pl = self.plot_collections(self.convergence_V_trace,6,green_white_brown)
        pl.set_linewidth(3)
        pl.set_alpha(0.6)
        plt.grid(True, lw=9, linestyle='-', color='w', alpha=0.3)
        line = ax.add_collection(pl)
        return

    @property 
    def convergence_V_trace(self):    
        if self.dimension%2 == 0:
            i = self.dimension/2
        else:
            i = self.dimension/2 + 1
        self.con_list = self.convergence_V[:,int(i)].tolist()
        final = self.scale_trace(self.con_list)
        final2 = 1.0 * final
        return final2

    @property
    def linear_convergence_array(self):
        conv_arr = np.asarray(self.convergence_V_trace)
        tile = np.tile(conv_arr,(self.dimension,1))
        final = np.transpose(tile)
        return final

    def linear_convergence_contour_plot(self):
        levels = np.arange(-1.01, 1.01, 0.01)
        vmax = np.max(self.linear_convergence_array)
        vmin = np.min(self.linear_convergence_array)
        ax.contourf(self.xx,self.yy,self.linear_convergence_array,levels=levels,vmin=vmin,cmap=divy, vmax=vmax,zorder=1, alpha=1)
        ax.contourf(self.xx,self.yy,self.linear_convergence_array,levels=levels,vmin=vmin,cmap=divy, vmax=vmax,zorder=1, alpha=1)
        return    

    def quiver_linear(self):
        #plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.rotation_U[self.skip()],self.rotation_V[self.skip()],color='k',alpha=0.6,zorder=10)
        plt.quiver(self.xx[self.skip2()],self.yy[self.skip2()],0,self.linear_convergence_array[self.skip2()],color='k',alpha=0.4,zorder=10)   # used with Azshear
        
    @property
    def convergence_V_full(self):
        self.derivative = []
        for r in range(0,self.dimension):
            self.this_row = []
            self.this_list = self.convergence_V[r].tolist()

            for i in range(0,self.dimension):
                if i == 0:
                    element = 0
                else:
                    element = self.this_list[i] - self.this_list[i - 1]
                self.this_row.append(element)

            self.this_row[0] = self.this_row[-1]
            smoothed = gaussian_filter1d(self.this_row, sigma=10)
            self.derivative.append(smoothed)
        #final = self.scale_trace(self.derivative)
        final = self.derivative
        return final

    def convergence_V_full_contour_plot(self):
        levels = np.arange(-2.01, 2.01, 0.2)
        vmax = np.max(self.convergence_V_full)
        vmin = np.min(self.convergence_V_full)
        ax.contourf(self.xx,self.yy,self.convergence_V_full,levels=levels,vmin=vmin,cmap=red_black_blue, vmax=vmax,zorder=1, alpha=1)
        ax.contourf(self.xx,self.yy,self.convergence_V_full,levels=levels,vmin=vmin,cmap=red_black_blue, vmax=vmax,zorder=1, alpha=1)
        return

    def convergence_speed_plot(self):

        levels = np.arange(-1, 1, 0.01)
        vmax = np.max(self.convergence_speed)
        vmin = np.min(self.convergence_speed)
        #norm = cm.colors.Normalize(vmax=vmax, vmin=vmin)
        #im = ax.imshow(self.convergence_speed, interpolation='nearest',cmap=white_brown, norm=norm)
        ax.contourf(self.xx,self.yy,self.convergence_speed,levels=levels,vmin=vmin,cmap=white_brown, vmax=vmax,zorder=1, alpha=1)
        ax.contourf(self.xx,self.yy,self.convergence_speed,levels=levels,vmin=vmin,cmap=white_brown, vmax=vmax,zorder=1, alpha=1)
        #fig.colorbar(im, ax=ax,shrink=0.75,ticks=[100])

    def convergence_V_plot(self):
        levels = np.arange(-1.001, 1.001, 0.1)
        vmax = np.max(self.convergence_V)
        vmin = np.min(self.convergence_V)
        #ax.contourf(self.xx,self.yy,self.convergence_V,levels=levels,vmin=vmin,cmap=green_white_brown, vmax=vmax,zorder=1, alpha=1)
        #ax.contourf(self.xx,self.yy,self.convergence_V,levels=levels,vmin=vmin,cmap=green_white_brown, vmax=vmax,zorder=1, alpha=1)
        ax.contour(self.xx,self.yy,self.convergence_V,vmin=vmin,cmap=green_white_brown, vmax=vmax,zorder=1, alpha=1)
        ax.contour(self.xx,self.yy,self.convergence_V,vmin=vmin,cmap=green_white_brown, vmax=vmax,zorder=1, alpha=1)

    def quiver_convergence_full(self):
        plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.convergence_U[self.skip()],self.convergence_V[self.skip()],color='k',alpha=0.8,zorder=10)
        #plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.convergence_U[self.skip()],self.convergence_V[self.skip()],color=self.convergence_speed,alpha=0.5,zorder=10)
        plt.grid(False)

    def quiver_convergence_U(self):
        #plt.quiver(self.xx[self.skip2()],self.yy[self.skip2()],self.convergence_U[self.skip2()],0,color='w',alpha=0.5,zorder=10)
        plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.convergence_U[self.skip()],0,color='k',alpha=0.5,zorder=10)
        plt.grid(False)

    def quiver_convergence_V(self):
        plt.quiver(self.xx[self.skip2()],self.yy[self.skip2()],0,self.convergence_V[self.skip2()],color='k',alpha=0.5,zorder=10)
        #plt.quiver(self.xx[self.skip()],self.yy[self.skip()],0,self.convergence_V[self.skip()],color='k',alpha=0.5,zorder=10)
        plt.grid(False)

    ##################################################################
    ###########################   Rotation   #########################
    ##################################################################

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
    def rotation_U(self):
        rot_U = self.rotation_inner()[0] + self.rotation_outer()[0]
        return rot_U

    @property    
    def rotation_V(self):
        rot_V = self.rotation_inner()[1] + self.rotation_outer()[1]
        return rot_V

    @property 
    def rotation_V_trace(self):    
        if self.dimension%2 == 0:
            i = self.dimension/2
        else:
            i = self.dimension/2 + 1
        this_list = self.rotation_V[int(i)].tolist()
        final = self.scale_trace(this_list)
        return final

    @property 
    def rotation_V_trace_all(self):
        V_traces = []
        for r in range(0,self.dimension):
            this_list = self.rotation_V[r].tolist()
            V_traces.append(this_list)
        return V_traces


    def rotation_V_trace_plot(self):
        pl = self.plot_collections(self.rotation_V_trace,2,green_gray_pink)
        pl.set_linewidth(6)
        pl.set_alpha(0.6)
        line = ax.add_collection(pl)
        return

    def rotation_V_indices(self):    
        rot_min = np.min(self.rotation_V_trace)
        rot_max = np.max(self.rotation_V_trace)
        for t in range(0,len(self.rotation_V_trace)):
            if self.rotation_V_trace[t] == rot_max:
                max_index = t
            elif self.rotation_V_trace[t] == rot_min:
                min_index = t
            else:
                pass
        return min_index,max_index


    def rotation_V_contour_plot(self):
        levels = np.arange(-8, 8, 1)
        vmax = np.max(self.rotation_V)
        vmin = np.min(self.rotation_V)
        ax.contourf(self.xx,self.yy,self.rotation_V,levels=levels,vmin=vmin,cmap=green_white_pink, vmax=vmax,zorder=1, alpha=0.8)
        ax.contourf(self.xx,self.yy,self.rotation_V,levels=levels,vmin=vmin,cmap=green_white_pink, vmax=vmax,zorder=1, alpha=0.8)
        return
    


    def quiver_rotation_full(self):
        #plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.rotation_U[self.skip()],self.rotation_V[self.skip()],color='k',alpha=0.6,zorder=10)
        plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.rotation_U[self.skip()],self.rotation_V[self.skip()],color='w',alpha=0.4,zorder=10)   # used with Azshear

    def quiver_rotation_V(self):
        plt.quiver(self.xx[self.skip2()],self.yy[self.skip2()],0,self.rotation_V[self.skip2()],color='k',alpha=0.5,zorder=10)

    ##################################################################
    ############ Rotation / Convergence / Translation  ###############
    ##################################################################

    @property
    def rotation_convergence_U(self):
        return self.rotation_U + self.convergence_U

    @property
    def rotation_convergence_V(self):
        return self.rotation_V + self.convergence_V  

    def quiver_rotation_convergence_full(self):
        plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.rotation_convergence_U[self.skip()],self.rotation_convergence_V[self.skip()],color='k',alpha=0.5,zorder=10)

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

    def quiver_full(self):
        #fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True)
        plt.quiver(self.xx[self.skip()],self.yy[self.skip()],self.full_U[self.skip()],self.rotation_convergence_V[self.skip()],color='k',alpha=0.5,zorder=10)
        plt.grid(False)


    @property
    def azshear_trace(self):
        self.derivative = []
        for i in range(0,len(self.rotation_V_trace)):
            if i == 0:
                element = 0
            else:
                element = self.rotation_V_trace[i] - self.rotation_V_trace[i - 1]
            self.derivative.append(element)
        self.derivative[0] = self.derivative[-1]
        smoothed = gaussian_filter1d(self.derivative, sigma=10)
        final = self.scale_trace(smoothed)
        return final

    @property
    def azshear_trace_plot(self):
        pl = self.plot_collections(self.azshear_trace,9,blue_black_red)        
        #vmax = np.max(self.azshear_trace) + 0.2
        #vmin = np.min(self.azshear_trace)
        pl.set_linewidth(6)
        pl.set_alpha(0.6)
        #pl.set_clim(vmin=vmin,vmax=vmax)
        line = ax.add_collection(pl)
        return

    @property
    def azshear_V_full(self):
        self.derivative = []
        for r in range(0,self.dimension):
            self.this_row = []
            self.this_list = self.rotation_V[r].tolist()

            for i in range(0,self.dimension):
                if i == 0:
                    element = 0
                else:
                    element = self.this_list[i] - self.this_list[i - 1]
                self.this_row.append(element)

            self.this_row[0] = self.this_row[-1]
            smoothed = gaussian_filter1d(self.this_row, sigma=10)
            self.derivative.append(smoothed)
        final = self.scale_trace(self.derivative)
        return final

    def azshear_V_full_contour_plot(self):
        levels = np.arange(-1.1, 1.1, 0.05)
        vmax = 1.5
        vmin = -1.5
        #vmax = np.max(self.azshear_V_full) + 0.2
        #vmin = np.min(self.azshear_V_full)
        ax.contourf(self.xx,self.yy,self.azshear_V_full,levels=levels,vmin=vmin,cmap=blue_black_red, vmax=vmax,zorder=1, alpha=1)
        ax.contourf(self.xx,self.yy,self.azshear_V_full,levels=levels,vmin=vmin,cmap=blue_black_red, vmax=vmax,zorder=1, alpha=1)
        return


    @property
    def divshear_V_full(self):
        self.derivative = []
        for r in range(0,self.dimension):
            self.this_column = []
            self.this_list = self.convergence_V[:,r].tolist()

            for i in range(0,self.dimension):
                if i == 0:
                    element = 0
                else:
                    element =  self.this_list[i - 1] - self.this_list[i]
                    #element = self.this_list[i - 1] - self.this_list[i]
                self.this_column.append(element)

            self.this_column[0] = self.this_column[-1]
            smoothed = gaussian_filter1d(self.this_column, sigma=10)
            self.derivative.append(smoothed)
        final = self.scale_trace(self.derivative)
        return final


    def divshear_V_full_contour_plot(self):
        levels = np.arange(-1.1, 1.1, 0.05)
        vmax = np.max(self.divshear_V_full)
        vmin = np.min(self.divshear_V_full)
        #vmax = np.max(self.divshear_V_full)
        #vmin = np.min(self.divshear_V_full)
        vmax = 1.5
        vmin = -1.5
        ax.contourf(self.xx,self.yy,self.divshear_V_full,levels=levels,vmin=vmin,cmap=blue_black_red, vmax=vmax,zorder=1, alpha=1)
        ax.contourf(self.xx,self.yy,self.divshear_V_full,levels=levels,vmin=vmin,cmap=blue_black_red, vmax=vmax,zorder=1, alpha=1)
        return

    @property
    def divshear_V_trace(self):
        self.derivative = []
        for i in range(0,len(self.convergence_V_trace)):
            if i == 0:
                element = 0
            else:
                element = self.convergence_V_trace[i] - self.convergence_V_trace[i - 1]
            self.derivative.append(element)
        self.derivative[0] = self.derivative[-1]
        final = self.scale_trace(self.derivative)
        return final

    def plot_collect(self,s,sig,cmap):
        norm = plt.Normalize(-1, 1)
        norm = plt.Normalize(np.min(s), np.max(s))
        smoothed = gaussian_filter1d(s, sigma=sig)
        points = np.array([self.x, smoothed]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap,norm=norm,alpha=0.4,zorder=10)
        lc.set_array(s)
        lc.set_linewidth(3)
        line = ax.add_collection(lc)
        return lc

    def plot_collections(self,s,sig,cmap):
        p_cols = self.plot_collect(s,sig,cmap)
        return p_cols


    def divshear_V_trace_plot(self):
        pl = self.plot_collections(self.divshear_V_trace,6,blue_black_red)
        pl.set_linewidth(6)
        pl.set_alpha(0.6)
        line = ax.add_collection(pl)
        return

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
# plot characteristics
plt.axis('scaled')
ax.set_yticks([0])
ax.set_xticks([10])
plt.grid(False)
ax.yaxis.set_ticklabels([])
ax.yaxis.set_ticks_position('none')
#ax.text(-0.14, -0.97, r'RADAR', fontsize=22,fontweight='bold',bbox=dict(facecolor='white', alpha=1),zorder=20) 

################################################################
# ---- quiver black 0.6 alpha ------

#test.rotation_V_contour_plot()
#test.quiver_rotation()
#fig_name='rotation_v_quiver.png'

################################################################
# rotation contour alpha = 0.8
#test.rotation_V_contour_plot()
#test.rotation_V_trace_plot()
#ax.grid(color='k', linestyle='--', alpha=0.4,linewidth=4)
#circle = plt.Circle((0, 0), test.rotmax_radius, color='k', alpha=0.3,linewidth=4,fill=False)
#ax.add_artist(circle)
#fig_name='circle_rotv_trace.png'

################################################################
#test.rotation_V_contour_plot()  # alpha set to 0.25
#test.rotation_V_trace_plot()    
#test.azshear_trace_plot()
#test.azshear_V_full_contour_plot()
#ax.grid(color='k', linestyle='--', alpha=0.4,linewidth=4)
circle = plt.Circle((0, 0), test.rotmax_radius, color='k', alpha=0.7,linewidth=4,linestyle='--',fill=False)
ax.add_artist(circle)
#fig_name='azshear_trace.png'
################################################################

#test.azshear_V_full_contour_plot()
#test.rotation_V_trace_plot()    
#test.quiver_rotation_full()
#fig_name='azshear_full.png'

################################################################
#test.convergence_speed_plot()
#test.quiver_convergence_full()
fig_name='conv_speed.png'
################################################################

test.linear_convergence_contour_plot()
test.quiver_linear()



#test.convergence_U_trace_plot()
#test.divshear_V_trace_plot()
#test.divshear_V_full_contour_plot()
#test.rotation_V_trace_plot()

#test.divshear_trace_plot()


ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
plt.tight_layout()
image_dst_path = os.path.join(image_dir,fig_name)
plt.savefig(image_dst_path,format='png',bbox_inches='tight')

