# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import numpy.ma as ma
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.pyplot as plt

class VortexGrid:
        def __init__(self,dimension=61,rotmax_fraction=0.4):
            self.dimension=dimension
            self.rotmax_fraction = rotmax_fraction
            x = np.linspace(-1,1,dimension)            
            y = np.linspace(-1,1,dimension)
            self.xx,self.yy = np.meshgrid(x,y)
            self.convergence = 0
            self.translation = 8
            self.distance = np.sqrt(self.xx**2 + self.yy**2)
            self.U_inner = np.ndarray([dimension,dimension])
            self.U_inner.fill(0)
            self.V_inner = np.ndarray([dimension,dimension])
            self.V_inner.fill(0)

            self.U_outer = np.ndarray([dimension,dimension])
            self.U_outer.fill(0)
            self.V_outer = np.ndarray([dimension,dimension])
            self.V_outer.fill(0)


            self.rot_max_radius = self.rotmax_fraction * self.distance.max()
            self.inner_radius_factor = self.distance/self.rot_max_radius
            self.outer_radius_factor = 1- (self.distance - self.rot_max_radius)/(self.distance.max() - self.rot_max_radius)
            self.sin_angle = 2*np.pi*(self.yy/self.distance)
            self.cos_angle = 2*np.pi*(self.xx/self.distance)

            self.rotation_u_inner = -1 * self.inner_radius_factor * self.sin_angle
            self.rotation_v_inner = self.inner_radius_factor * self.cos_angle
            self.rotation_u_outer = -1 * self.outer_radius_factor * self.sin_angle
            self.rotation_v_outer = self.outer_radius_factor * self.cos_angle
            
            self.convergence_u_inner = self.inner_radius_factor * self.convergence * self.cos_angle
            self.convergence_v_inner = self.inner_radius_factor * self.convergence * self.sin_angle
            self.convergence_u_outer = self.outer_radius_factor * self.convergence * self.cos_angle
            self.convergence_v_outer = self.outer_radius_factor * self.convergence * self.sin_angle

#
            self.U_inner = self.rotation_u_inner + self.convergence_u_inner + self.translation
            self.V_inner = self.rotation_v_inner + self.convergence_v_inner

            self.U_inner_prefill = ma.masked_array(self.U_inner,self.distance > self.rot_max_radius)
            self.V_inner_prefill = ma.masked_array(self.V_inner,self.distance > self.rot_max_radius)
            self.U_inner_filled = self.U_inner_prefill.filled(fill_value=0)
            self.V_inner_filled = self.V_inner_prefill.filled(fill_value=0)

            self.U_outer = self.rotation_u_outer + self.convergence_u_outer + self.translation
            self.V_outer = self.rotation_v_outer + self.convergence_v_outer

            self.U_outer_prefill = ma.masked_array(self.U_outer,self.distance <= self.rot_max_radius)
            self.V_outer_prefill = ma.masked_array(self.V_outer,self.distance <= self.rot_max_radius)
            self.U_outer_filled = self.U_outer_prefill.filled(fill_value=0)
            self.V_outer_filled = self.V_outer_prefill.filled(fill_value=0)


            

            self.U = self.U_inner_filled + self.U_outer_filled
            self.V = self.V_inner_filled + self.V_outer_filled
            self.magnitude = np.sqrt(self.U**2 + self.V**2)

test = VortexGrid()
skip = (slice(None, None, 4), slice(None, None, 4))
fig, axes = plt.subplots(1,1,figsize=(8,8))
U_ms = test.U * units.meter / units.second
V_ms = test.V * units.meter / units.second
vorticity = mpcalc.vorticity(U_ms, V_ms, 2 * units.meter, 2 * units.meter)
plt.contour(test.xx,test.yy,test.magnitude,color='k',zorder=1)
plt.quiver(test.xx[skip],test.yy[skip],test.U[skip],test.V[skip],test.magnitude[skip])
#plt.streamplot(test.xx, test.yy, test.U, test.V, color=test.magnitude, density=0.2, zorder=10)
plt.axis('scaled')
#plt.hist2d(test.xx,test.yy,test.angle)


            