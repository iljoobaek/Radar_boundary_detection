#
# Copyright (c) 2018, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# azimuth-range FFT heatmap - 2D plot
#

import os, sys, copy, math
import cv2 as cv
import random as rng
rng.seed(12345)
import tkinter as tk
from tkinter import filedialog


#try:
    
import numpy as np

import scipy.interpolate as spi

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.widgets as wgt
import matplotlib.patches as pat
from matplotlib.widgets import Slider, Button

from plot import *
import plot
'''
except ImportError:
    print("import error")
    sys.exit(3)
'''
# --- Constants --- #

COLORMAP_MAX = 3000
COLOR_THRESHOLD = 1200

# ------------------------------------------------

# colormap maximum
cm_max = COLORMAP_MAX

# Object detection will be turned-on if contour is true
# Current algorithm for object detection is color thresholding + Canny edge detection to get contour
# And use decision-based method on the contours.
threshold = COLOR_THRESHOLD
contour = False

# ----- Helper functions for buttons and sliders ----- #
def cm_max_update(val):
    if contour:
        return
    global cm_max
    cm_max = val

def threshold_update(val):
    global threshold
    threshold = val

def contour_update(event):
    global contour, cm_max, threshold
    contour = not contour
    if not contour:
        cm_max = COLORMAP_MAX
    else:
        threshold = COLOR_THRESHOLD

def forward_update(event):
    if read_serial == 'serial':
        return
    plot.frame_count += 100

def backward_update(event):
    if read_serial == 'serial':
        return
    if plot.frame_count <= 100:
        return
    plot.frame_count -= 100

# ------------------------------------------------ #

def valid_boundary(contour_poly):
    origin = (199.5 , 0)
    distance_max = 0.0      # unit is index
    distance_min = 1000.0   # unit is index
    angle_max = -180.0
    angle_min = 180.0
    # x_max = 0 
    # x_min = 1000
    # y_max = 0
    # y_min = 1000
    for point in contour_poly:
        #print(point)
        dist = np.linalg.norm(point - origin)
        if dist > distance_max:
            distance_max = dist
        if dist < distance_min:
            distance_min = dist

        angle = np.angle((point[0][0] - origin[0]) + point[0][1] * 1j , deg=True)
        if angle > angle_max:
            angle_max = angle
        if angle < angle_min:
            angle_min = angle
        # if point[0][0] > x_max:
        #     x_max = point[0][0]
        # if point[0][0] < x_min:
        #     x_min = point[0][0]
        # if point[0][1] > y_max:
        #     y_max = point[0][1]
        # if point[0][1] < y_min:
        #     y_min = point[0][1]
    
    image_res = range_res * range_bins / grid_res
    variance = (distance_max - distance_min) * image_res  # unit is meter
    
    angle_span = angle_max - angle_min
    #print("angle_max, angle_min: " + str(angle_max) + "," + str(angle_min))
    distance = image_res * (distance_max + distance_min) / 2
    criteria = 0.1 / (2 * distance * math.pi) * 360
    if criteria < 8:
        criteria = 8
    #print("distance: " + str(distance) + " criteria: " + str(criteria) + " degrees")

    # distance variance shouldn't be larger than 0.8 m
    if variance > 0.8:
        return False
    # angle span should be larger
    if angle_span < criteria:
        return False
    # objects within 80 cm are discarded, since the housing is giving near-field noise.
    if distance < 0.8:
        return False 

    # print("x_max   x_min   y_max   y_min")
    # print(str(x_max) + "     " + str(x_min) + "     " + str(y_max) + "     " + str(y_min))
    # print("=== rectangle center x,y = " + str(image_res * ((x_max + x_min) / 2 - origin[0])) 
    #                     + "," + str(image_res * ((y_max + y_min) / 2 - origin[1])) + " ===")
    # print("distance_max,distance_min = " + str(distance_max * image_res) + "," + str(distance_min * image_res))
    # print("image_res: " + str(image_res) + " variance: " + str(variance))
    # print("")
    
    return True

def contour_rectangle(zi):
    zi_copy = np.uint8(zi)
    canny_output = cv.Canny(zi_copy, 100, 100 * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    boundary_or_not = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 0.01, True)
        #print("shape of contour_poly[" + str(i) + "] " + str(contours_poly[i].shape))
        #print(contours_poly[i])
        boundRect[i] = cv.boundingRect(contours_poly[i])
        boundary_or_not[i] = valid_boundary(contours_poly[i])

    print(boundary_or_not)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 4), dtype=np.uint8)

    for i in range(len(contours)):
        if boundary_or_not[i]:
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(drawing, contours_poly, i, color)
            cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), 
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)   
    return drawing

def update(data):

    global bshowcm_max, cm_max, threshold
    bshowcm_max.label.set_text("CM_MAX: " + str(int(cm_max)) + "\nThreshold: " + str(int(threshold))
                        + "\nAngle Bins: " + str(angle_bins)
                            + "\nScope: " + str(scope))
    if not 'azimuth' in data or len(data['azimuth']) != range_bins * tx_azimuth_antennas * rx_antennas * 2:
        return

    a = data['azimuth']
    timer_start = time.time()
    a = np.array([a[i] + 1j * a[i+1] for i in range(0, len(a), 2)])
    a = np.reshape(a, (range_bins, tx_azimuth_antennas * rx_antennas))
    a = np.fft.fft(a, angle_bins)
    print ("it took %fs for fft"%(time.time() - timer_start))
     
    timer_start = time.time()
    a = np.abs(a)
    a = np.fft.fftshift(a, axes=(1,))  # put left to center, put center to right       
    a = a[:,1:]  # cut off first angle bin
    print ("it took %fs for fftshift"%(time.time() - timer_start))

    timer_start = time.time()
    a = a[:, scope_start:scope_end].ravel()
    zi = spi.griddata((x, y), a, (xi, yi), method='linear')
    #zi = a
    zi = np.fliplr(zi)
    print ("it took %fs for griddata and flip"%(time.time() - timer_start))

    if contour:
        timer_start = time.time()
        cm_max = 255
        ret, zi = cv.threshold(zi,threshold,cm_max,cv.THRESH_BINARY)
        drawing = contour_rectangle(zi)
        cm.set_array(drawing[::-1,::-1,0] + zi[::-1,::-1])
        #cm.set_array(drawing[::-1,::-1,0])
        print ("it took %fs for creating contour"%(time.time() - timer_start))
    else:
        timer_start = time.time()
        cm.set_array(zi[::-1,::-1])  # rotate 180 degrees
        print ("it took %fs for rotate 180 degrees"%(time.time() - timer_start))

    if heat_mode[heat_choice] == 'rel':
        cm.autoscale()  # reset colormap
        #return zi
    elif heat_mode[heat_choice] == 'abs':
        cm.set_clim(0, cm_max)  # reset colormap
        #return zi


if __name__ == "__main__":


    # plot.frame_count = 200
    # print("frame_count: " + str(plot.frame_count))

    if len(sys.argv[1:]) != 9:
        print('Usage: {} {}'.format(sys.argv[0].split(os.sep)[-1], 
            '<num_tx_azim_antenna> <num_rx_antenna> <num_range_bin> <num_angular_bin> <range_bin> <range_bias> <scope> <trunc> <read/serial>'))
        sys.exit(1)
        
    #try:

    # use mouse left button to toggle relative colormap or absolute colormap.
    heat_mode, heat_choice = ('rel', 'abs'), 1
    
    tx_azimuth_antennas = int(float(sys.argv[1]))
    rx_antennas = int(float(sys.argv[2]))
    
    range_bins = int(float(sys.argv[3]))
    angle_bins = int(float(sys.argv[4]))

    # 1st 2: phasors' real and imaginary part
    # 2nd 2: 2 bytes each for real and imaginary part
    plot.PAYLOAD_SIZE_DEFAULT = int((range_bins / float(sys.argv[8])) * tx_azimuth_antennas * rx_antennas * 2 * 2)
    
    range_res = float(sys.argv[5])
    range_bias = float(sys.argv[6])

    scope = float(sys.argv[7])
    scope_start = 0.5 - scope
    scope_end = 0.5 + scope

    # truncating the data to reduce the data set size for gridding
    # maximum range is reduced at the same time
    plot.PAYLOAD_TRUNC = float(sys.argv[8])

    read_serial = sys.argv[9]
    logpath = ""
    if read_serial == 'read':
        root = tk.Tk()
        root.withdraw()
        logpath = filedialog.askopenfilename()
        root.destroy()
    
    
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
    scope_start = round(x.shape[1] * scope_start)
    scope_end = round(x.shape[1] * scope_end)
    x = x[:, scope_start:scope_end].ravel()
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
                        
    ax.set_title('Azimuth-Range FFT Heatmap: Right', fontsize=16)
    ax.set_xlabel('Lateral distance along [m]')
    ax.set_ylabel('Longitudinal distance along [m]')

    ax.plot([0, 0], [0, range_depth], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, -range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, +range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)    

    for i in range(1, int(range_depth)):
        ax.add_patch(pat.Arc((0, 0), width=i*2, height=i*2, angle=90, 
                        theta1=-90, theta2=90, color='white', linewidth=0.5, linestyle=':', zorder=1))

    #fig.canvas.mpl_connect('button_press_event', onclick)

    # choose colors here: https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    axcolor = 'mistyrose'

    # --- Set the position of the buttons and sliders --- #
    axcm_max = plt.axes([0.2, 0.001, 0.65, 0.02], facecolor=axcolor)
    scm_max = Slider(axcm_max, 'cm_max', 0, 10000, valinit = cm_max, valstep=500, color='brown')

    axthreshold = plt.axes([0.2, 0.021, 0.65, 0.02], facecolor=axcolor)
    sthreshold = Slider(axthreshold, 'threshold', 500, 4000, valinit = threshold, valstep=100, color='brown')

    axcontour = plt.axes([0.1, 0.04, 0.1, 0.02])
    bcontour = Button(axcontour, 'Contour', color='lightblue', hovercolor='0.9')

    axforward = plt.axes([0.25, 0.04, 0.3, 0.02])
    bforward = Button(axforward, 'Forward(100 frames)', color='lightblue', hovercolor='0.9')

    axbackward = plt.axes([0.6, 0.04, 0.3, 0.02])
    bbackward = Button(axbackward, 'Backward(100 frames)', color='lightblue', hovercolor='0.9')

    axshowcm_max = plt.axes([0.8, 0.8, 0.17, 0.15], facecolor=axcolor)
    bshowcm_max = Button(axshowcm_max, "CM_MAX: " + str(int(cm_max)) 
                    + "\nThreshold: " + str(int(threshold))
                        + "\nAngle Bins: " + str(angle_bins)
                            + "\nScope: " + str(scope), color='lightblue', hovercolor='0.9')
    bshowcm_max.label.set_fontsize(24)

    # --- Register callbacks of the sliders --- #
    scm_max.on_changed(cm_max_update)
    sthreshold.on_changed(threshold_update)

    # --- Register callbacks of the buttons --- #
    bcontour.on_clicked(contour_update)
    bforward.on_clicked(forward_update)
    bbackward.on_clicked(backward_update)

    # --- Start the core of application based on serial or replay --- #

    if read_serial == 'serial':
        start_plot(fig, ax, update)
    elif read_serial == 'read':
        replay_plot(fig, ax, update, logpath)

    ''' 
    except Exception:
        sys.exit(2)
    '''
