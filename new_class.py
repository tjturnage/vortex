# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:55:03 2020

@author: thomas.turnage
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

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


        self.dimension=dimension
        self.rotmax_fraction = rotmax_fraction
        self.convergence = convergence
        self.azshear = []
        self.azshear_surge = []
        self.x = np.linspace(-1,1,dimension)        
        self.y = np.linspace(-1,1,dimension)
        self.xx,self.yy = np.meshgrid(self.x,self.y)
        # set up x and y grids and mesh them into a 2d grid

            
        # calculate a distance from origin for each grid point


    @property
    def distance(self):
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

    @property
    def outer_radius_factor(self):
        return 1/(self.distance/self.rotmax_radius)


    @property   
    def convergence_inner(self):
        u = self.inner_radius_factor * self.convergence * self.cos_angle
        v = self.inner_radius_factor * self.convergence * self.sin_angle
        u_ma = ma.masked_array(u,self.distance > self.rotmax_radius)
        u_ma_fill = u_ma.filled(fill_value=0)  
        v_ma = ma.masked_array(v,self.distance > self.rotmax_radius)
        v_ma_fill = v_ma.filled(fill_value=0) 
        return u_ma_fill, v_ma_fill  


    @property   
    def convergence_outer(self):
        u = self.outer_radius_factor * self.convergence * self.cos_angle
        v = self.outer_radius_factor * self.convergence * self.sin_angle
        u_ma = ma.masked_array(u,self.distance > self.rotmax_radius)
        u_ma_fill = u_ma.filled(fill_value=0)  
        v_ma = ma.masked_array(v,self.distance > self.rotmax_radius)
        v_ma_fill = v_ma.filled(fill_value=0) 
        return u_ma_fill, v_ma_fill  
  
    @property   
    def rotation_inner(self):
        u = (-1 * self.inner_radius_factor * self.sin_angle)
        v = (self.inner_radius_factor * self.cos_angle)
        u_ma = ma.masked_array(u,self.distance > self.rotmax_radius)
        u_ma_fill = u_ma.filled(fill_value=0)  
        v_ma = ma.masked_array(v,self.distance > self.rotmax_radius)
        v_ma_fill = v_ma.filled(fill_value=0) 
        return u_ma_fill, v_ma_fill

    @property   
    def rotation_outer(self):
        u = (-1 * self.outer_radius_factor * self.sin_angle)
        v = (self.outer_radius_factor * self.cos_angle)
        u_ma = ma.masked_array(u,self.distance <= self.rotmax_radius)
        u_ma_fill = u_ma.filled(fill_value=0)  
        v_ma = ma.masked_array(v,self.distance <= self.rotmax_radius)
        v_ma_fill = v_ma.filled(fill_value=0) 
        return u_ma_fill, v_ma_fill

    @property   
    def rot_U(self):
        rot_U = self.rotation_inner[0] + self.rotation_outer[0]
        return rot_U

    @property    
    def rot_V(self):
        rot_V = self.rotation_inner[1] + self.rotation_outer[1]
        return rot_V

    def translation_factor(self):
        self.V_max = np.max(self.rot_V)
        self.V_min = np.min(self.rot_V)
        self.U_max = np.max(self.rot_U)
        self.magnitude_max = np.max(np.sqrt(np.square(self.V_max) + np.square(self.U_max)))
        new_translation = self.translation * self.magnitude_max
        return new_translation

    def rot_V_trace(self):    
        if self.dimension%2 == 0:
            i = self.dimension/2
        else:
            i = self.dimension + 1
        rot_V_trace = self.rot_V[int(i)]
        return rot_V_trace


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
        