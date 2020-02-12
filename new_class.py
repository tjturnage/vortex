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
from scipy.ndimage.filters import gaussian_filter1d
from custom_cmaps import plts

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
        self.skip_val = int(self.dimension/5)
        self.rotmax_fraction = rotmax_fraction
        self.convergence = convergence
        self.azshear = []
        self.azshear_surge = []
        self.derivative = []
        self.x = np.linspace(-1,1,dimension)        
        self.y = np.linspace(-1,1,dimension)
        self.xx,self.yy = np.meshgrid(self.x,self.y)
        # set up x and y grids and mesh them into a 2d grid

            
        # calculate a distance from origin for each grid point

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
    def cartesian_grid(self):
        xx,yy = np.meshgrid(self.x,self.y)
        a = np.ndarray([self.dimension,self.dimension])
        a.fill(0)
        return a

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
        u_fill = self.ma_outer(u)
        v_fill = self.ma_outer(v)        
        return u_fill, v_fill  


    @property   
    def convergence_outer(self):
        u = self.outer_radius_factor * self.convergence * self.cos_angle
        v = self.outer_radius_factor * self.convergence * self.sin_angle
        u_fill = self.ma_outer(u)
        v_fill = self.ma_outer(v)        
        return u_fill, v_fill  

    @property   
    def rot_U(self):
        rot_U = self.rotation_inner()[0] + self.rotation_outer()[0]
        return rot_U

    @property    
    def rot_V(self):
        rot_V = self.rotation_inner()[1] + self.rotation_outer()[1]
        return rot_V

    def translation_factor(self):
        self.V_max = np.max(self.rot_V)
        self.V_min = np.min(self.rot_V)
        self.U_max = np.max(self.rot_U)
        self.magnitude_max = np.max(np.sqrt(np.square(self.V_max) + np.square(self.U_max)))
        new_translation = self.translation * self.magnitude_max
        return new_translation


#    def masked_inner(self,grid):
#        masked = ma.masked_array(grid,self.distance > self.rotmax_radius)        
#        filled = masked.filled(fill_value=0)
#        return filled

#    def ma_inner(self,grid):
#        ma_grid = self.masked_inner(grid)
#        return ma_grid
#
#    def masked_outer(self,grid):
#        masked = ma.masked_array(grid,self.distance <= self.rotmax_radius)        
#        filled = masked.filled(fill_value=0)
#        return filled
#

#
#    def ma_outer(self,grid):
#        ma_grid = self.masked_outer(grid)
#        return ma_grid

    def rot_V_trace(self):    
        if self.dimension%2 == 0:
            i = self.dimension/2
        else:
            i = self.dimension + 1
        rot_V_trace = self.rot_V[int(i)]
        return rot_V_trace

    def rot_V_arr(self,rot_V_trace):
        rot_V_arr = self.rot_V_trace()
        return rot_V_arr

    def azshear(self):
        if self.dimension%2 == 0:
            i = self.dimension/2
        else:
            i = self.dimension + 1
        rot_V_trace = self.rot_V[int(i)]

        for i in range(0,len(rot_V_trace())):
            if i == 0:
                element = 0
            else:
                element = rot_V_trace()[i] - rot_V_trace()[i - 1]
            
            self.derivative.append(element)
        
        self.derivative[0] = self.derivative[-1]
        return self.derivative

#    def array_derivative(self,arr_input_1d):
#        derivative = []
#        for i in range(0,len(arr_input_1d)):
#            if i == 0:
#                element = 0
#            else:
#                element = arr_input_1d[i] - arr_input_1d[i - 1]
#            
#            derivative.append(element)
#        
#        derivative[0] = derivative[-1]
#        return derivative
#
#    def arr_deriv(self,arr_input_1d):
#        deriv = self.array_derivative(arr_input_1d)
#        return deriv
#
#
#    def azshear(self,arr_input_1d):
#        self.arr_input_1d = self.rot_V_trace()
#        print(self.arr_input_1d)
#        azshear = self.arr_deriv(self.rot_V_trace())
#        return azshear

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


    def quiver_skip(self):
        skip = (slice(None, None, self.skip_val), slice(None, None, self.skip_val))
        # plot quivers with substantial alpha
        return skip



    def plot_collections(self,s):
        smoothed = gaussian_filter1d(s, sigma=8)
        points = np.array([self.x, smoothed]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plts['just_gray']['cmap'],norm=norm,alpha=0.4,zorder=10)
        lc.set_array(y)
        lc.set_linewidth(3)
        line = ax.add_collection(lc)



test = None
test = VortexGrid2()
#azshear = test.arr_deriv(test.rot_V_trace())
 
