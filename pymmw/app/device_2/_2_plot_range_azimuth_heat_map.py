#
# Copyright (c) 2018, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# azimuth-range FFT heatmap - 2D plot
#

import os, sys

#try:
    
import numpy as np

import scipy.interpolate as spi

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.widgets as wgt
import matplotlib.patches as pat

from _2_plot import *
'''
except ImportError:
    print("import error")
    sys.exit(3)
'''
# np.set_printoptions(precision=2)
# np.set_printoptions(suppress=True)

# ------------------------------------------------

def onclick(event):
    
    if event.button == 3:  # toggle scale of data
        pass
    
    if event.button == 1:  # toggle color range of heatmap
        global heat_choice
        heat_choice += 1
        heat_choice %= len(heat_mode)
        
    return (event.xdata, event.ydata)


def update(data):
    
    if not 'azimuth' in data or len(data['azimuth']) != range_bins * tx_azimuth_antennas * rx_antennas * 2:
        return

    a = data['azimuth']
    #timer_start = time.time()
    a = np.array([a[i] + 1j * a[i+1] for i in range(0, len(a), 2)])
    a = np.reshape(a, (range_bins, tx_azimuth_antennas * rx_antennas))
    a = np.fft.fft(a, angle_bins)
    #print ("it took %fs for fft"%(time.time() - timer_start))
     
    #timer_start = time.time()
    a = np.abs(a)
    a = np.fft.fftshift(a, axes=(1,))  # put left to center, put center to right       
    a = a[:,1:]  # cut off first angle bin
    #print ("it took %fs for fftshift"%(time.time() - timer_start))

    #timer_start = time.time()
    a = a[:, scope_start:scope_end].ravel()
    zi = spi.griddata((x, y), a, (xi, yi), method='cubic')
    #zi = a
    zi = np.fliplr(zi)
    #print ("it took %fs for griddata and flip"%(time.time() - timer_start))
    
    #timer_start = time.time()
    cm.set_array(zi[::-1,::-1])  # rotate 180 degrees
    #print ("it took %fs for rotate 180 degrees"%(time.time() - timer_start))
    #cm.set_array(zi)
    # cm.set_array(zi.ravel())
    if heat_mode[heat_choice] == 'rel':
        cm.autoscale()  # reset colormap
        #zmin, zmax = np.nanmin(zi), np.nanmax(zi)
        #print("zmin: " + str(zmin) + " zmax: " + str(zmax))
        #zmax = 1000
        #cm.set_clim(zmin, zmax)  # reset colormap
    elif heat_mode[heat_choice] == 'abs':
        cm.set_clim(0, 6000)  # reset colormap


if __name__ == "__main__":

    if len(sys.argv[1:]) != 6:
        print('Usage: {} {}'.format(sys.argv[0].split(os.sep)[-1], '<num_tx_azim_antenna> <num_rx_antenna> <num_range_bin> <num_angular_bin> <range_bin> <range_bias>'))
        sys.exit(1)
        
    #try:

    heat_mode, heat_choice = ('rel', 'abs'), 1
    scope = 0.1
    scope_start = 0.5 - scope
    scope_end = 0.5 + scope
    
    tx_azimuth_antennas = int(float(sys.argv[1]))
    rx_antennas = int(float(sys.argv[2]))
    
    range_bins = int(float(sys.argv[3]))
    angle_bins = int(float(sys.argv[4]))
    
    range_res = float(sys.argv[5])
    range_bias = float(sys.argv[6])
    
    # ---
        
    t = np.array(range(-angle_bins//2 + 1, angle_bins//2)) * (2 / angle_bins)
    t = np.arcsin(t) # t * ((1 + np.sqrt(5)) / 2)
    r = np.array(range(range_bins)) * range_res

    range_depth = range_bins * range_res
    range_width, grid_res = range_depth / 2, 400
    range_width = range_width * scope * 4
    
    xi = np.linspace(-range_width, range_width, grid_res)
    yi = np.linspace(0, range_depth, grid_res)
    xi, yi = np.meshgrid(xi, yi)

    x = np.array([r]).T * np.sin(t)
    y = np.array([r]).T * np.cos(t)
    #print( "shape: " + str(x.shape))
    scope_start = round(x.shape[1] * scope_start)
    scope_end = round(x.shape[1] * scope_end)
    #print( "scope: " + str(scope_start) + " " + str(scope_end))
    x = x[:, scope_start:scope_end].ravel()
    print( "shape: " + str(x.shape))
    y = y[:, scope_start:scope_end].ravel()
    y = y - range_bias

    # ---

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)  # rows, cols, idx
    
    move_figure(fig, (0 + 45*3, 0 + 45*3))
    
    plt.tight_layout(pad=2)
    
    cm = ax.imshow(((0,)*grid_res,) * grid_res, cmap=plt.cm.jet, extent=[-range_width, +range_width, 0, range_depth], alpha=0.95)
    

    cursor = wgt.Cursor(ax, useblit=True, color='white', linewidth=1)
    
    fig.canvas.set_window_title('...')
                        
    ax.set_title('Azimuth-Range FFT Heatmap: Left', fontsize=16)
    ax.set_xlabel('Lateral distance along [m]')
    ax.set_ylabel('Longitudinal distance along [m]')

    ax.plot([0, 0], [0, range_depth], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, -range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, +range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)    

    for i in range(1, int(range_depth)):
        ax.add_patch(pat.Arc((0, 0), width=i*2, height=i*2, angle=90, theta1=-90, theta2=90, color='white', linewidth=0.5, linestyle=':', zorder=1))

    fig.canvas.mpl_connect('button_press_event', onclick)
    
    start_plot(fig, ax, update)

    ''' 
    except Exception:
        sys.exit(2)
    '''
