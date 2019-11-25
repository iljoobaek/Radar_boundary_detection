#
# Copyright (c) 2018, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# azimuth-range FFT heatmap - 2D plot
#

import os, sys, copy, math
from math import sqrt
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

import socket

# --- Constants --- #

COLORMAP_MAX = 3000
COLOR_THRESHOLD = 700

port = "/tmp/radarPacket"
server_address = ('localhost', 17699)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
firstRun = True

# ------------------------------------------------

# colormap maximum
cm_max = COLORMAP_MAX

# Object detection will be turned-on if contour is true
threshold = COLOR_THRESHOLD
contour = True

# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ----- Helper function for generating ground truth ----- #
# ----- log the mouse click ----- #
# Mouse left click indicates a detected object. Right click implies not found.
mouse_in_heatmap = False
def onclick(event):
    if not mouse_in_heatmap:
        print("[on-click] mouse not in heatmap. Not logged.")
        return
    
    try:
        print('[on-click] %s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            ('double' if event.dblclick else 'single', event.button,
            event.x, event.y, event.xdata, event.ydata))
        if event.button == 3:
            flush_ground_truth(plot.frame_count, -1)
        else:
            point = np.zeros((2,1))
            point[0] = event.xdata
            point[1] = event.ydata
            origin = np.zeros((2,1))
            dist = np.linalg.norm(point - origin)
            flush_ground_truth(plot.frame_count, dist)
    except:
        print("[on-click] exception: probably scrolled")

def enter_axes(event):
    if type(event.inaxes) == type(ax): 
        print("in heatmap")
        global mouse_in_heatmap
        mouse_in_heatmap = True
        event.inaxes.patch.set_facecolor('white')
        event.canvas.draw()

def leave_axes(event):
    global mouse_in_heatmap
    mouse_in_heatmap = False
    event.inaxes.patch.set_facecolor('yellow')
    event.canvas.draw()

# ----- flush the data into ground_truth.txt ----- #
# distance will be -1 if no object detected.
last_frame_count = 0
def flush_ground_truth(frame_count, distance):
    global last_frame_count
    if frame_count == last_frame_count:
        print("[flush_ground_truth] click too fast - last: %d this: %d" % (last_frame_count, frame_count))
        print("[flush_ground_truth] skip!")
        return
    print("[flush_ground_truth] frame_count: %d distance: %f" % (frame_count, distance))
    #print(os.path.basename(logpath).strip(".dat"))
    ground_truth_path = "DATA/ground_truth_" + os.path.basename(logpath).strip(".dat") + ".txt"
    ground_truth_path = "DATA/ground_truth_fake.txt"
    
    # if the script is running temporary to generate ground truth of temporary objects,
    # change the file name to temporary_gtound_truth
    if read_serial == 'temporary':
        ground_truth_path = ground_truth_path.replace('ground_truth', 'temporary_ground_truth')

    #ground_truth_path = "DATA/ground_truth_" + os.path.basename(logpath).strip(".dat") + ".txt"
    with open(ground_truth_path, "a") as f:
        data = str(frame_count) + ',' + ("%.5f" % distance) + '\n'
        f.write(data)
    f.close()
    last_frame_count = frame_count
    print("[flush_ground_truth] data flushed!")
    return

# ----- Read ground truth data from text file ----- #
ground_truth = {}
def read_ground_truth():
    #ground_truth_path = "DATA/ground_truth_" + os.path.basename(logpath).strip(".dat") + ".txt"
    ground_truth_path = "DATA/ground_truth_fake.txt"
    with open(ground_truth_path, "r") as f:
        for line in f:
            ground_truth[int(line.split(',')[0])] = float(line.split(',')[1])
    return

# ----- flush the detected distance into test.txt ----- #
def flush_test(frame_count, distance):
    test_path = "DATA/test_" + os.path.basename(logpath).strip(".dat") + ".txt"
    with open(test_path, "a") as f:
        data = str(frame_count) + ',' + ("%.5f" % distance) + '\n'
        f.write(data)
    f.close()
    return

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
    plot.frame_count -= 1

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

def calculate_length(angle_span, distance):
    return distance * 2 * math.pi * angle_span / 360.

def generate_data_msg(length, distance):
    return ','.join([str(length)[0:14],str(distance)[0:14]])

def send_msg(msg):
    print(msg)
    s.send(msg.encode('ascii'))
    #sock.send(msg.encode('ascii'))
    #sock.send("\n".encode('ascii'))
    
# ------------------------------------------------ #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ----- Helper function - Angle Span ----- #
def angle_span_interp(distance):
    # Currently:
    # Distance: 0m   Span: at least 30 degrees
    # Distance: 15m  Span: at least 6 degrees
    # (interpolations between 0m and 15m)
    return (30 + (30 - 6) / (0 - 15) * distance)

# ----- Helper function - first step: generating possible objects ----- #
# get the distribution of the distances of the contour points.
# then return the closer side of the contour
def generate_distance_index(distances):
    mean = np.mean(distances)
    std = np.std(distances)
    return mean - 2 * std, mean

# core of first step:
# make decision based on angle span
# return the distance of the boundary
def valid_boundary(contour_poly):
    global firstRun
    if firstRun:
        #s.connect(port)
        sock.connect(server_address)
        firstRun = False
    origin = (199.5 , 0)
    distance_max = 0.0      # unit is index
    distance_min = 1000.0   # unit is index
    angle_max = -180.0
    angle_min = 180.0
    distances = []
    for point in contour_poly:
        dist = np.linalg.norm(point - origin)
        if dist > distance_max:
            distance_max = dist
        if dist < distance_min:
            distance_min = dist
        distances.append(dist)

        angle = np.angle((point[0][0] - origin[0]) + point[0][1] * 1j , deg=True)
        if angle > angle_max:
            angle_max = angle
        if angle < angle_min:
            angle_min = angle
    
    image_res = range_res * range_bins / grid_res
    variance = (distance_max - distance_min) * image_res  # unit is meter
    
    angle_span = angle_max - angle_min
    
    # get the distance of the boundary
    distance, distance_middle = generate_distance_index(distances)
    distance = image_res * distance
    distance_middle = image_res * distance_middle

    # get the velocity of it
    velocity = doppler_lookup(distance_middle)

    # get the angle span criteria with the distance
    criteria = angle_span_interp(distance)

    # angle span should be larger
    if angle_span < criteria:
        return False , distance, angle_span, velocity

    # objects within 80 cm are discarded, since the housing is giving near-field noise.
    if distance < 0.5555555555555555555555555555555555555555555555555555555:
        return False , distance, angle_span, velocity
    
    return True , distance, angle_span, velocity

# ----- Helper function - second step: making decisions ----- #
def box_distance(box):
    origin = (199.5, 0)
    close_dist = 1000
    far_dist = 1000
    image_res = range_res * range_bins / grid_res
    # store the left bottom point as close value
    point = np.asarray([box[0],box[1]])
    close_dist = np.linalg.norm(point - origin) * image_res
    # store the right bottom point as close value if it's closer to origin
    point = np.asarray([box[0]+box[2], box[1]])
    if close_dist > np.linalg.norm(point - origin) * image_res:
        close_dist = np.linalg.norm(point - origin) * image_res

    # store the left top point as close value
    point = np.asarray([box[0],box[1]+box[3]])
    far_dist = np.linalg.norm(point - origin) * image_res
    # store the right top point as close value if it's closer to origin
    point = np.asarray([box[0]+box[2],box[1]+box[3]])
    if far_dist > np.linalg.norm(point - origin) * image_res:
        far_dist = np.linalg.norm(point - origin) * image_res
    
    if close_dist > far_dist:
        tmp = close_dist
        close_dist = far_dist
        far_dist = tmp

    return close_dist, far_dist

def noise_removal(boundary_or_not, distance, contours_poly, zi_copy):
    object_index = []
    global tracker_box, tracker_mismatch
    # get the list of the valid objects
    match, mismatch = 0,0
    for i in range(len(boundary_or_not)):
        if boundary_or_not[i] :
            # remove the object that are not around the place the tracker reports
            if len(tracker_box) != 0:
                success, box = tracker.update(zi_copy)
                if success:
                    close_dist, far_dist = box_distance(box)
                    print("close_dist: %f far_dist: %f distance[i]: %f" % (close_dist, far_dist, distance[i]))
                    if close_dist - distance[i] > 0.5 or distance[i] - far_dist > 0.5:
                        boundary_or_not[i] = False
                        mismatch += 1
                        continue
                    else:
                        match += 1
            object_index.append(i)

    if mismatch > match:
        tracker_mismatch += 1
        print(">>> tracker_mismatch: " + str(tracker_mismatch))
    
    if len(object_index) == 0:
        return boundary_or_not

    # cluster the valid objects with the distance
    index_cluster = cluster_by_distance(object_index, distance)

    outlier = -1
    only_clusters = True
    last_cluster = -1
    for i in range(len(index_cluster)):
        if len(index_cluster[i]) == 1:
            only_clusters = False
            if i > outlier:
                outlier = i
        else:
            if i > last_cluster:
                last_cluster = i
    
    # first flip all to false
    for i in range(len(boundary_or_not)):
        boundary_or_not[i] = False        

    # then flip only one to true to report the object
    if not only_clusters:
        if outlier > last_cluster:
            boundary_or_not[index_cluster[outlier][0]] = True
        else:
            boundary_or_not[index_cluster[last_cluster][-1]] = True
    else:
        boundary_or_not[object_index[-1]] = True

    return boundary_or_not

# ----- Helper function for clustering: mean of list ----- #
def mean(lst):
    n = float(len(lst))
    mean = sum(lst) / n
    # stdev = sqrt((sum(x*x for x in lst) / n) - (mean * mean)) 
    return mean

# ----- Helper function for clustering ----- #
# ----- generating clusters by checking the distance to mean ----- #
def process(distance, object_index, criteria=1):
    dist_cluster = []
    index_cluster = []
    for i in range(len(distance)):
        try:
            object_index.index(i)
        except:
            continue

        if len(dist_cluster) < 1:    # the first two values are going directly in
            dist_cluster.append(distance[i])
            index_cluster.append(i)
            continue

        cluster_mean = mean(dist_cluster)
        if abs(cluster_mean - distance[i]) > criteria:    # check the "distance"
            yield index_cluster
            dist_cluster[:] = []    # reset cluster to the empty list
            index_cluster[:] = []    # reset cluster to the empty list

        dist_cluster.append(distance[i])
        index_cluster.append(i)
    yield index_cluster

# ----- clustering by distance ----- #
def cluster_by_distance(object_index, distance):
    ret = []
    for cluster in process(distance, object_index):
        ret.append(cluster.copy())
        #ret.append(cluster)
    return ret

# ----- Main function for object detection: generating contour & rectangles ----- #
tracker_box = []
tracker = cv.TrackerKCF_create()
tracker_failure = 0
tracker_mismatch = 0
ret_dist_rolling = []
def contour_rectangle(zi):
    zi_copy = np.uint8(zi)
    contours, _ = cv.findContours(zi_copy, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    boundary_or_not = [None]*len(contours)
    distance = [None]*len(contours)
    length = [None]*len(contours)
    velocity = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 0.01, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        # Karun get data here!
        boundary_or_not[i], distance[i], angle_span, velocity[i] = valid_boundary(contours_poly[i])
        length[i] = calculate_length(distance[i], angle_span)
        # Karun get data here!
    
    timestamp = str(int(time.time() * 1000000))
    if read_serial == 'read' or read_serial == 'test':
        timestamp = get_timestamp(frame_count)

    if any(boundary_or_not):
        send_msg("Start:" + timestamp + "Len:" + "{0:0=3d}".format(np.sum(boundary_or_not)+1))
    else:
        send_msg("Start:" + timestamp + "Len:000")
    boundaryIndices = np.where(boundary_or_not)[0]
    #send_msg("Possible results:")
    [send_msg(generate_data_msg(y, x)) for y, x in zip(np.take(length, boundaryIndices).tolist(), np.take(distance, boundaryIndices).tolist())]

    boundary_or_not = noise_removal(boundary_or_not, distance, contours_poly, zi_copy)
    
    drawing = np.zeros((zi_copy.shape[0], zi_copy.shape[1], 4), dtype=np.uint8)
    labels = np.zeros((zi_copy.shape[0], zi_copy.shape[1], 4), dtype=np.uint8)

    ret_dist = -1
    
    global tracker_box, tracker, tracker_failure, tracker_mismatch
    for i in range(len(contours)):
        if boundary_or_not[i]:
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(drawing, contours_poly, i, color)
            cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), 
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)

            if len(tracker_box) == 0:
                tracker_box = boundRect[i]
                tracker.init(zi_copy, tracker_box)

            cv.putText(labels, ("d: %.4f" % distance[i]), 
                            (grid_res - int(boundRect[i][0] - 10), grid_res - int(boundRect[i][1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv.putText(labels, ("v: %.4f" % velocity[i]), 
                            (grid_res - int(boundRect[i][0] - 25), grid_res - int(boundRect[i][1]) - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)   
            
            # Karun get data here!
            ret_dist = distance[i]
            ret_length = length[i]
            ret_velocity = velocity[i]
            # Karun get data here!
            #send_msg("Most possible result:")
            send_msg(generate_data_msg(length[i], distance[i]))
            timestamp = str(int(time.time() * 1000000))
            if read_serial == 'read' or read_serial == 'test':
                timestamp = get_timestamp(frame_count)
            send_msg("PacketFinish:" + timestamp)

    global ret_dist_rolling
    if len(tracker_box) != 0:
        sucess, box = tracker.update(zi_copy)
        if sucess:
            tracker_failure = 0
            print(">>> tracker success!")
            cv.rectangle(drawing, (int(box[0]), int(box[1])), 
                    (int(box[0]+box[2]), int(box[1]+box[3])), (180,180,180), 8)
            print(">>> ret_dist: " + str(ret_dist))
            
            if len(ret_dist_rolling) != 0 and np.mean(ret_dist_rolling) - ret_dist > 1:
                ret_dist_rolling = []
            
            ret_dist_rolling.append(ret_dist)
            ret_dist = np.mean(ret_dist_rolling)
            print(">>> ret_dist(rolling): " + str(ret_dist))
        else:
            tracker_failure += 1
            print(">>> tracker_failure: " + str(tracker_failure))

    if tracker_failure > 8 or tracker_mismatch > 4:
        print(">>> tracker reset! tracker_failure: %d tracker_mismatch: %d" %(tracker_failure, tracker_mismatch))
        tracker_box = []
        tracker_failure = 0
        tracker_mismatch = 0
        tracker = cv.TrackerKCF_create()
        ret_dist_rolling = []
    
    return drawing, labels, ret_dist


# ----- doppler initialization ----- #
doppler_vec = []
def doppler_init(data):
    global doppler_vec
    a = np.array(data['doppler'])
    a = np.reshape(a, (range_bins, doppler_bins))
    shift = 12
    a = np.concatenate((a[:,shift:16],a[:,0:shift]) , axis=1)
    doppler_vec = a[:,1:].T
    print("[dopplet init] shape of doppler vec: " + str(doppler_vec.shape))

# ----- doppler velocity look up -----#
def doppler_lookup(distance):
    index = int(distance / range_res)
    #index = range_bins - index
    # print("index of %.4f is %d" % (distance, index))
    try:
        profile = doppler_vec[:,index]
    except:
        profile = doppler_vec[:,range_bins - 1]
    # print("===== profile =====")
    # print(profile)
    # print("===================")
    # print("[doppler_lookup] index of max: " + str(np.argmax(profile)))
    velocity = doppler_res * (np.argmax(profile) - 7)
    # print("[doppler_lookup] velocity: " + str(velocity))
    # print("===================")
    return velocity

# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ----- Helper function to put ground_truth on plot ----- #
curb_arc_patch = pat.Arc((0,0), 1, 1)
def update_ground_truth():
    global curb_arc_patch
    curb_arc_patch.remove()
    ground_truth_distance = 0.001
    try:
        ground_truth_distance = ground_truth[plot.frame_count]
    except:
        print("frame_count not in ground truth")
    curb_arc_patch = pat.Arc((0, 0), width=ground_truth_distance*2, height=ground_truth_distance*2, angle=90, 
                        theta1=-30, theta2=30, color='magenta', linewidth=3, linestyle=':', zorder=1)
    ax.add_patch(curb_arc_patch)
    return

# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ----- Main function for updating the plot ----- #
def update(data, msg=''):
    if msg is not '':
        send_msg(msg)
        return
    
    global bshowcm_max, cm_max, threshold
    bshowcm_max.label.set_text("CM_MAX: " + str(int(cm_max)) + "\nThreshold: " + str(int(threshold))
                        + "\nAngle Bins: " + str(angle_bins)
                            + "\nScope: " + str(scope))
    if not 'azimuth' in data or len(data['azimuth']) != range_bins * tx_azimuth_antennas * rx_antennas * 2:
        print("azimuth fail! " + str(len(data['azimuth'])))
        return
    
    if not 'doppler' in data or len(data['doppler']) != range_bins * doppler_bins:
        print("doppler fail! " + str(len(data['doppler'])))
        return
    
    timer_start = time.time()
    doppler_init(data)
    print ("it took %fs for doppler_init"%(time.time() - timer_start))
    
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

    update_ground_truth()

    if contour:
        timer_start = time.time()
        cm_max = 255
        ret, zi = cv.threshold(zi,threshold,cm_max,cv.THRESH_BINARY)
        drawing, labels, ret_dist = contour_rectangle(zi)
        cm.set_array(drawing[::-1,::-1,0] + zi[::-1,::-1] + labels[:,:,0])
        if plot.flush_test_data:
            flush_test(plot.frame_count, ret_dist)
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

# ----- Helper function to update angle span interpolation ----- #
def add_angle_span_arc():
    for dist in range(1 , int(range_depth)):
        angle_span = angle_span_interp(dist)
        angle_span_arc = pat.Arc((0, 0), width=dist*2, height=dist*2, angle=90, 
                        theta1=-angle_span/2, theta2=angle_span/2, color='yellow', linewidth=3, linestyle=':', zorder=1)
        ax.add_patch(angle_span_arc)
    
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ---------------------------------------------------------- #
# ----- Application Entry ----- #
if __name__ == "__main__":


    # plot.frame_count = 200
    # print("frame_count: " + str(plot.frame_count))

    if len(sys.argv[1:]) < 9:
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
    doppler_bins = 16
    doppler_res = 0.1252

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
    if read_serial == 'read' or read_serial == 'ground_truth' or read_serial == 'temporary':
        root = tk.Tk()
        root.withdraw()
        logpath = filedialog.askopenfilename()
        root.destroy()
    
    if read_serial == 'test':
        plot.flush_test_data = True
        logpath = sys.argv[10]

    try:
        read_ground_truth()
    except:
        print("No ground truth yet!")
    

    
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

    ax.add_patch(curb_arc_patch)
    #add_angle_span_arc()
    curb_label = ax.text(-range_width * 0.9, range_width * 0.25, "Curb", color='magenta', fontsize='xx-large')
    tracker_label = ax.text(-range_width * 0.9, range_width * 0.4, "Tracker", color='gold', fontsize='xx-large')
    #fig.canvas.mpl_connect('button_press_event', onclick)

    # choose colors here: https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    axcolor = 'mistyrose'

    # --- Set the position of the buttons and sliders --- #
    axcm_max = plt.axes([0.2, 0.001, 0.65, 0.02], facecolor=axcolor)
    scm_max = Slider(axcm_max, 'cm_max', 0, 10000, valinit = cm_max, valstep=500, color='brown')

    axthreshold = plt.axes([0.2, 0.021, 0.65, 0.02], facecolor=axcolor)
    sthreshold = Slider(axthreshold, 'threshold', 200, 4000, valinit = threshold, valstep=100, color='brown')

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
    elif read_serial == 'read' or read_serial == 'test':
        replay_plot(fig, ax, update, logpath)
    elif read_serial == 'ground_truth' or read_serial == 'temporary':
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('axes_enter_event', enter_axes)
        fig.canvas.mpl_connect('axes_leave_event', leave_axes)
        replay_plot(fig, ax, update, logpath, False)
    s.close()
    ''' 
    except Exception:
        sys.exit(2)
    '''
