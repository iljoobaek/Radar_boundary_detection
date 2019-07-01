#
# Copyright (c) 2018, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# abstract plot support
#

import sys, time, threading, json, queue, serial

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import art3d

# ------------------------------------------------

def set_aspect_equal_3d(ax):  # axis have to be equal 

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)

    plot_radius = max([abs(lim - mean_) for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean)) for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def move_figure(fig, xy):
    backend = mpl.get_backend()
    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry("+%d+%d" % xy)
    elif backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition(xy)
    else:  # QT and GTK
        fig.canvas.manager.window.move(*xy)



# ----- Some numbers about payload manipulation ----- #
# Now this only support packet with 1 TLV and azimuth heat map specific.
# 

PAYLOAD_START = 44
PAYLOAD_TRUNC = 0.75
PAYLOAD_SIZE = int(8192 * PAYLOAD_TRUNC)
PACKET_SIZE = PAYLOAD_START + PAYLOAD_SIZE

# ----- Name of device ----- #

device_name = '/dev/tty.usbmodem000000004'

# ----- Magic Word ----- #
magic_word = "0201040306050807"

# ----- from Serial ----- #
def from_serial(filename, chunksize=PACKET_SIZE-8):
    #with open(filename, "rb") as f:
    #with serial.Serial(device_name, 961200, timeout=1) as f:
    with filename as f:
        pattern = ""
        count = 0
        while True:
            c = f.read(1)
            if count > 5000:
                print("magic word search out of control")
                return
            if not c:
                break
            if pattern == "" or len(pattern) < 16:
                pattern += c.hex()
            else:
                pattern = pattern[2:16]
                pattern += c.hex()

            if pattern == magic_word:
                #print("match: " + pattern)
                for it in range(0,16,2):
                    b = pattern[it] + pattern[it+1]
                    yield int(b)
                
                chunk = f.read(chunksize)
                for b in chunk:
                    yield b
                
                return
            count += 1

# ----- The header of the packet ----- #
header = {
    'VERSION' : 8,
    'PACKETLEN' : 12,
    'PLATFORM' : 16,
    'FRAMENUMBER' : 20,
    'CPUCYCLE' : 24,
    'DETECTOBJ' : 28,
    'NUMTLV' : 32
}

# ----- The header of the TLV ----- #
tlvheader = {
    'TLVTYPE' : 36,
    'TLVLEN' : 40
}


# ----- The frame here is the frame displayed on GUI, not chirp frame ----- #
frame_count = 0

# ----- Global variables for storing data ----- #
# bytevec: store all of the packet parsed.
# datavec: store all the phasor tuples
# datamap: to meet the requirement of update() in plot_range_azimuth_heat_map.py

bytevec = []
datavec = []
datamap = {
    'azimuth' : datavec
}

# ----- Verify byte vec ----- #
def verify_vec(vec):
    if len(vec) != PACKET_SIZE:
        print("[serial-on-the-fly] len of serialvec: " + str(len(vec)))
        return False
    """ 
    pattern = ""
    for i in range(0,8):
        pattern += vec[i].hex()
    if not pattern == magic_word :
        print("[verify] pattern not matched: " + pattern)
        return False
    

    count = 8
    for i in range(8,len(vec)):
        pattern = pattern[2:16]
        pattern += vec[i].hex()
        if pattern == magic_word:
            print("[verify] shall not match twice! at: " + str(count))
            return False
        count += 1 """

    return True

total_count = 0
fail_count = 0
# ----- Read from serial on-the-fly ----- #
def serial_on_the_fly(fig, q):
    global total_count, fail_count
    f = serial.Serial(device_name, 961200, timeout=1)
    while q.alive:
        #try:
            """ try:
                f.inWaiting()
            except:
                print("reconnect...")
                f = serial.Serial(device_name, 961200, timeout=1)
                time.sleep(1)
                continue """
            serialvec = []
            for b in from_serial(f):
                serialvec.append(bytes([b]))

            if not verify_vec(serialvec):
                fail_count += 1
                total_count += 1
                print("[serial-on-the-fly] failed: " + str(round(100 * fail_count/total_count, 2)) + " % failed")
                continue
            total_count += 1
            q.put(serialvec)
            #print("[serial-on-the-fly] serialvec enqueued! size: " + str(len(serialvec)))
            #time.sleep(0.001)
            
        #except:
        #    q.alive = False

# ----- Process bytevec to datavec ----- #
def collect_data(start,end):
    global datavec
    datavec.clear()
    count = 0
    first , second = 0 , 0

    for byteindex in range(start,end,2):
        intbyte = bytevec[byteindex] + bytevec[byteindex + 1]
        byteint = int.from_bytes(intbyte, byteorder='little', signed=True)
        if count % 2 == 0:
            first = byteint
        if count % 2 == 1:
            second = byteint
            datavec.append(second)
            datavec.append(first)
        count += 1

    #print("len of datavec: " + str(len(datavec)))
    #print("count: " + str(count))
    #print("first 4 numbers(swapped): %d %d %d %d" % (datavec[0], datavec[1], datavec[2], datavec[3]))

queueMax = 1

# ----- Update the plot ----- #
def update_plot(fig, q, func, tu):
    
    count = 0
    
    while q.alive:
        global frame_count
        global bytevec

        
        # 
        # When having a complete packet, turn that into datavec and pass it back to update()
        # Stops when the rest of file size is not long enough for another packet
        #
        if not q.empty():
            bytevec = q.get()
            q.task_done()
            start = PAYLOAD_START
            end = PAYLOAD_START + PAYLOAD_SIZE
            collect_data(start,end)
            func(datamap)
            #print("[update_plot] len of datamap['azimuth']: " + str(len(datamap['azimuth'])))
            #print("[update_plot] queue size: " + str(q.qsize()))

            frame_count += 1
        else:
            #print("serial thread alive? " + str(tu.is_alive()))
            #serial_on_the_fly(fig, q)
            #print("[update_plot] wait for data...")
            time.sleep(1e-6)
            continue
            

        try:
            fig.canvas.draw_idle()
            fig.canvas.set_window_title("frame: " + str(count))
            count += 1
            time.sleep(1e-6)
        except:
            q.alive = False


def start_plot(fig, ax, func):
    
    plt.show(block=False)

    bytevecQueue = queue.Queue(queueMax)

    bytevecQueue.alive = True

    tu = threading.Thread(target=serial_on_the_fly, args=(fig,bytevecQueue))
    tu.daemon = True
    tu.start()

    threading.Thread(target=update_plot, args=(fig, bytevecQueue, func, tu)).start()
    
    
    
    plt.show(block=True)
    bytevecQueue.alive = False
