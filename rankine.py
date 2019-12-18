# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 09:01:37 2019

@author: tjtur
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import pathlib
import os
import shutil
import re

def build_html(image_dir):


    page_title = 'Vortex'
    loop_description = 'Vortices<br>'

    
    # following file has to be copied into image directory
    # available at https://github.com/tjturnage/resources for download
    # if not manually putting into image directory, will need to note its location and execute the following
    # two commands
    js_src = 'C:/data/scripts/resources/hanis_min.js'
    #js_src = '/data/scripts/resources/hanis_min.js'
    shutil.copyfile(js_src,os.path.join(image_dir,'hanis_min.js'))
    
    # there will be an index.html file created in the image directory to be subsequently opened in an internet browser
    index_path = os.path.join(image_dir,'index.html')
    these_files = os.listdir(image_dir)
    
    # build list of image filenames
    file_str = ''
    for f in (these_files):
        if (re.search('png',f) is not None) or (re.search('png',f) is not None):
            file_str = file_str + f + ', ' 
    
    #trim unwanted characters from string
    file_str = file_str[0:-2]
    
    # first part of html code
    html_1 = '<!doctype html>\
    <html>\
    <head>\
    <meta charset="utf-8">\
    <title>' + page_title + '</title>\
    <script type="text/javascript" src="hanis_min.js"></script>\
    <style>\
    body {\
    	background-color: #434343;\
    	color: white;\
    	font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;\
    	font-size: 12px;\
    	text-align: center;\
    }\
    #container {\
    	position: relative;\
    	width: 700px;\
     	margin: 0 auto 0 auto;\
    }\
    #hanis {\
    	background-color: #AEAEAE;\
    }\
    a, a:link, a:visited {\
    	color: lightblue;\
    }\
    a:hover {\
    	color: lightgreen;\
    }\
    </style>\
    </head>\
    \
    <body onload="HAniS.setup(\'filenames = '
    
    """
    Example of filenames list...
    ('filenames = 'HRRRMW_prec_radar_000.png, HRRRMW_prec_radar_001.png, HRRRMW_prec_radar_002.png, HRRRMW_prec_radar_003.png, HRRRMW_prec_radar_004.png, HRRRMW_prec_radar_005.png, HRRRMW_prec_radar_006.png, HRRRMW_prec_radar_007.png, HRRRMW_prec_radar_008.png, HRRRMW_prec_radar_009.png, HRRRMW_prec_radar_010.png, HRRRMW_prec_radar_011.png, HRRRMW_prec_radar_012.png, HRRRMW_prec_radar_013.png, HRRRMW_prec_radar_014.png, HRRRMW_prec_radar_015.png, HRRRMW_prec_radar_016.png, HRRRMW_prec_radar_017.png, HRRRMW_prec_radar_018.png\ncontrols = startstop, speed, step, looprock, zoom\ncontrols_style = display:flex;flex-flow:row;\nbuttons_style = flex:auto;margin:2px;cursor:pointer;\nbottom_controls = toggle\ntoggle_size = 8,8,2\ndwell = 100\npause = 1000','hanis')">
    """
    
    file_line_end = '\\ncontrols = startstop, speed, step, looprock, zoom\\ncontrols_style = display:flex;flex-flow:row;\\nbuttons_style = flex:auto;margin:2px;cursor:pointer;\\nbottom_controls = toggle\\ntoggle_size = 8,8,2\\ndwell = 100\\npause = 1000\',\'hanis\')">\
      <div id="container">\
        <div id="hanis"></div>\
        <p>' + loop_description + '\n\
          Velocity products developed by CIMMS (<a href="https://cimms.ou.edu/" target_="blank">https://cimms.ou.edu/</a>) / NSSL (<a href="https://www.nssl.noaa.gov/" target="_blank">https://www.nssl.noaa.gov/</a>) group and are experimental<br>Animation javascript developed by Tom Whittaker (<a href="http://www.ssec.wisc.edu/hanis/">http://www.ssec.wisc.edu/hanis/</a>)\
        </p>\
      </div>\
    </body>\
    </html>'
    
    full_html = html_1 + file_str + file_line_end
    
    f = open(index_path,'w')
    f.write(full_html)
    f.close()
    return

thetas = np.linspace(0,2*np.pi,18)

radii_1 = np.linspace(0.1,1,10,endpoint=True)
theta1, r1 = np.meshgrid(thetas, radii_1)

radii_2 = np.linspace(1.1,2,10,endpoint=True)
theta2, r2 = np.meshgrid(thetas, radii_2)

conv_ratio = random.uniform(0.25,1)
speed_to_rotation_ratio = random.uniform(0,2)

u = np.arange(0.0,2.2,0.2)
c = np.arange(0.0,1.1,0.1)

for uu in u:
    for cc in c:

        rot_max = 1
        #conv_max = -1 * conv_ratio
        conv_max = -1 * cc
        #u = speed_to_rotation_ratio
        u = uu        
        
        f = plt.figure(figsize=(10,12))
        ax = f.add_subplot(111, polar=True)
        
        dt1 = r1**2 * rot_max
        dr1 = conv_max * r1
        U1 = dr1 * np.cos(theta1) - dt1 * np.sin (theta1)
        V1 = dr1 * np.sin(theta1) + dt1 * np.cos(theta1)
        magnitude1 = np.sqrt((U1 + u)**2 + V1**2)
        color1 = magnitude1
        ax.quiver(theta1, r1, U1+u, V1,color1, cmap=plt.cm.jet)
        
        
        
        dt2 = rot_max * np.power(2-r2,2)
        dt2 = rot_max * (1/r2**4)
        #dt2 =  rot_max * (2-r2)
        
        dr2 = conv_max * (2-r2)
        dr2 = conv_max * (1/r2**3)
        #dr2 = conv_max * np.power(2-r2,0.5)
        U2 = dr2 * np.cos(theta2) - dt2 * np.sin (theta2)
        V2 = dr2 * np.sin(theta2) + dt2 * np.cos(theta2) 
        magnitude2 = np.sqrt((U2+u)**2 + V2**2)
        color2 = magnitude2
        ax.quiver(theta2, r2, U2+u, V2,color2, cmap=plt.cm.jet)
        
        
        
        
        #ax.contour(magnitude)
        #ax.streamplot(theta, r, dr * np.cos(theta) - dt * np.sin (theta) + val, dr * np.sin(theta) + dt * np.cos(theta) + val)
        #ax.streamplot(theta2, r2, dr2 * np.cos(theta2) - dt2 * np.sin (theta2) + val2, dr2 * np.sin(theta2) + dt2 * np.cos(theta2) + val2)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
        title_text1 = 'Surface Winds From a Random Tornado Vortex!\n\nMaximum Rotational Velocity = 1\n\n'
        title_text2 = f'Vortex Eastward Velocity = {uu:0.2f}        Maximum Convergence Velocity = {cc:0.2f}'
        
        ax.set_title(title_text1 + title_text2)
        
        #image_dst_path = "/var/www/html/images/vortex/vortex.png"
        fname = f'{int(10*uu):02}u_{int(10*cc):03}conv_vortex.png'        
        #fname = str(int(10*cc)) + str(int(10*uu)) + '_vortex.png'
        image_dst_dir = "C:/data/vps/images/vortex/combos/"
        plt.savefig(os.path.join(image_dst_dir,fname),format="png",bbox_inches="tight")
        plt.close()
#ax.quiver(theta, r, dr * np.cos(theta) - dt * np.sin (theta), dr * np.sin(theta) + dt * np.cos(theta))
build_html(image_dst_dir)
#ax.quiver(theta, r, u * val , 0)

"""
,bbox_inches='tight'

x = np.linspace(-1,1,30, endpoint=True)
y = np.linspace(-1,1,30, endpoint=True)


X, Y = np.meshgrid(x, y)

f = plt.figure(figsize=(10,10))
ax = f.add_subplot(111)
#plt.plot(x,y, marker='.', color='k', linestyle='none')

#ax.plot(X,Y, marker='.', color='k', linestyle='none')
#U = x*np.sin(2*np.pi*(Y**2 + X**2))
#V = y*(2*np.pi)
#V = y*np.cos(y/x)
#V = x*np.cos(2*np.pi*(Y**2 + X**2))
radius = np.sqrt(X**2 + Y**2)
theta = 2*np.pi*(Y/X)
pi = np.pi/4
two_pi = 2 * np.pi
magnitude = np.sqrt(X**2 + Y**2)
Urot = -Y*np.cos(pi*X)**2
Vrot = X*np.cos(pi*Y)
Uconv = -X/2
Vconv = -Y/2

U = 2*Urot + Uconv + 2
V = 2*Vrot + Vconv
#V = X**2
#V = np.sqrt(X**2 + Y**2)
plt.quiver(x,y,U,V, color='k')
#plt.streamplot(x,y,U,V)
#sax.plot(x,y,D,D)

"""