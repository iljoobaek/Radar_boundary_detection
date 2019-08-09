#
# Copyright (c) 2018, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# abstract plot support
#

import sys, time, threading, json, queue, serial, datetime, cv2, os, signal

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import art3d

# ------------------------------------------------------------------------------------------- #

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

# ------------------------------------------------------------------------------------------- #

flush_test_data = False

# ----- Some numbers about payload manipulation ----- #
# Now this only support packet with 1 TLV and azimuth heat map specific.
# 

PAYLOAD_START = 44
PAYLOAD_TRUNC = 0.375
PAYLOAD_SIZE_DEFAULT = 8192
PAYLOAD_SIZE = int(PAYLOAD_SIZE_DEFAULT * PAYLOAD_TRUNC)
PACKET_SIZE = PAYLOAD_START + PAYLOAD_SIZE
PACKET_SIZE_DEFAULT = PAYLOAD_START + PAYLOAD_SIZE_DEFAULT
PACKET_SIZE_DEFAULT_DOPPLER = PACKET_SIZE_DEFAULT + PAYLOAD_SIZE_DEFAULT

# ----- Name of device ----- #

device_name = '/dev/tty.usbmodem000000004'

# ----- File name of log ----- #

filename = "/Users/wcchung/OneDrive/Main/17699/TI SDK/Radar_boundary_detection/pymmw/app/DATA/binary-2019-07-02-16-57-54.dat"

# ----- Magic Word ----- #
magic_word = "0201040306050807"

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
# datavec: store all the phasor tuples. Every real & imaginary part consists of two bytes.
# datamap: to meet the requirement of update() in plot_range_azimuth_heat_map.py

bytevec = []
datavec = []
dopplervec = []
datamap = {
    'azimuth' : datavec,
    'doppler' : dopplervec
}

# ------------------------------------------------------------------------------------------- #

# ----- Thread: Read data from serial ----- #
# ----- Serial reading helper function ----- #
def from_serial(filename, chunksize=PACKET_SIZE-8):
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

# ----- Thread: Read data from serial ----- #
# ----- Verify byte vec helper function ----- #
def verify_vec(vec):
    global PACKET_SIZE
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
# ----- Thread: Read data from serial ----- #
# ----- Read from serial on-the-fly ----- #
def serial_on_the_fly(fig, q, loggingQueue):
    global total_count, fail_count, PACKET_SIZE
    f = serial.Serial(device_name, 961200, timeout=1)
    while q.alive and loggingQueue.alive:
        serialvec = []
        for b in from_serial(f, PACKET_SIZE_DEFAULT - 8):
            serialvec.append(bytes([b]))

        if not verify_vec(serialvec[:PACKET_SIZE]):
            fail_count += 1
            total_count += 1
            print("[serial-on-the-fly] failed: " + str(round(100 * fail_count/total_count, 2)) + " % failed")
            continue
        total_count += 1
        q.put(serialvec[:PACKET_SIZE])
        loggingQueue.put(serialvec)
        print("[serial-on-the-fly] serialvec enqueued! size: " + str(len(serialvec)))
        time.sleep(1e-6) # yield the interest of scheduler

# ------------------------------------------------------------------------------------------- #
# ----- Thread: Save to local filesystem ----- #
# ----- Get bytevec or datavec from queues ----- #
def background_Logging(bytevecQueue, datavecQueue):
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    while bytevecQueue.alive and datavecQueue.alive:
        if not bytevecQueue.empty():
            binary_data = bytevecQueue.get()
            bytevecQueue.task_done()
            write_byte_to_log(binary_data, timestamp)
        # if not datavecQueue.empty():
        #     integer_data = datavecQueue.get()
        #     write_int_to_log(integer_data, timestamp)
        time.sleep(1e-6) # yield the interest of scheduler
    
def write_byte_to_log(binary_data, timestamp):
    filename = "DATA/binary-" + timestamp + ".dat"
    with open(filename, "ab") as f:
        for byte in binary_data:
            f.write(byte)
        f.close()

def write_int_to_log(integer_data, timestamp):
    filename = "DATA/integer-" + timestamp + ".dat"
    with open(filename, "a") as f:
        for integer in integer_data:
            f.write(str(integer))
            f.write(',')
        f.write('\n')
        f.close()


# ------------------------------------------------------------------------------------------- #
# ----- Thread: updating the plot ----- #
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

def collect_doppler(start,end):
    global dopplervec
    dopplervec.clear()

    for byteindex in range(start,end,2):
        intbyte = bytevec[byteindex] + bytevec[byteindex + 1]
        byteint = int.from_bytes(intbyte, byteorder='little', signed=True)
        dopplervec.append(byteint)

# ----- Thread: updating the plot ----- #
# ----- Update the plot ----- #
def update_plot(fig, ax, q, func, loggingQueue):
    
    #count = 0
    
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
            #loggingQueue.put(datavec)
            func(datamap)
            #print("[update_plot] len of datamap['azimuth']: " + str(len(datamap['azimuth'])))
            #print("[update_plot] queue size: " + str(q.qsize()))

            frame_count += 1
        else:
            time.sleep(1e-6) # yield the interest of scheduler
            continue

        try:
            fig.canvas.draw_idle()
            ax.set_title('Azimuth-Range FFT Heatmap: ' + str(frame_count) + ' frames', fontsize=48)
            fig.canvas.set_window_title("frame: " + str(frame_count))
            # print(" >>>>>>>>>>>>> frame_count : " + str(frame_count) + " <<<<<<<<<")
            # print(" >>>>>>>>>>>>> frame_count : " + str(frame_count) + " <<<<<<<<<")
            # print(" >>>>>>>>>>>>> frame_count : " + str(frame_count) + " <<<<<<<<<")
            # print(" >>>>>>>>>>>>> frame_count : " + str(frame_count) + " <<<<<<<<<")
            
            time.sleep(1e-6) # yield the interest of scheduler
        except:
            q.alive = False


# ------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------- #
# ----- Spawning threads: data from serial ----- #
queueMax = 1
def start_plot(fig, ax, func):
    
    global PAYLOAD_SIZE, PACKET_SIZE, PAYLOAD_TRUNC
    PAYLOAD_SIZE = int(PAYLOAD_SIZE_DEFAULT * PAYLOAD_TRUNC)
    PACKET_SIZE = PAYLOAD_START + PAYLOAD_SIZE
    print("PAYLOAD_SIZE_DEFAULT: " + str(PAYLOAD_SIZE_DEFAULT))
    print("PAYLOAD_TRUNC: " + str(PAYLOAD_TRUNC))
    print("PAYLOAD_SIZE: " + str(PAYLOAD_SIZE))
    print("PACKET_SIZE: " + str(PACKET_SIZE))

    plt.show(block=False)

    bytevecQueue = queue.Queue(queueMax)
    bytevecQueue.alive = True

    byteveclogging = queue.Queue(1)
    dataveclogging = queue.Queue(1)
    byteveclogging.alive = True
    dataveclogging.alive = True

    # The first thread is to get data from serial and put it in byte vector queue
    serial_thread = threading.Thread(target=serial_on_the_fly, args=(fig,bytevecQueue, byteveclogging))
    serial_thread.daemon = True
    serial_thread.start()

    # The second thread will log the bytevec and datavec with the starting timestamp
    logging_thread = threading.Thread(target=background_Logging, args=(byteveclogging, dataveclogging))
    logging_thread.daemon = True
    logging_thread.start()

    # The third thread will get data from queue when available.
    # Then construct complex number array, put it in datamap and pass it back to main script for plotting.
    threading.Thread(target=update_plot, args=(fig, ax, bytevecQueue, func, dataveclogging)).start()
    
    
    plt.show(block=True)
    bytevecQueue.alive = False
    byteveclogging.alive = False
    dataveclogging.alive = False

# ------------------------------------------------------------------------------------------- #
# ----- Spawning threads: data from file ----- #
queueMax = 1
def replay_plot(fig, ax, func, filepath, ground_truth=False):
    global filename
    filename = filepath
    print("\n***** filename: " + filename + " *****\n")

    global PAYLOAD_SIZE, PACKET_SIZE, PAYLOAD_TRUNC
    PAYLOAD_SIZE = int(PAYLOAD_SIZE_DEFAULT * PAYLOAD_TRUNC)
    # -----
    PAYLOAD_SIZE = 7680
    # -----
    PACKET_SIZE = PAYLOAD_START + PAYLOAD_SIZE
    print("PAYLOAD_SIZE_DEFAULT: " + str(PAYLOAD_SIZE_DEFAULT))
    print("PAYLOAD_TRUNC: " + str(PAYLOAD_TRUNC))
    print("PAYLOAD_SIZE: " + str(PAYLOAD_SIZE))
    print("PACKET_SIZE: " + str(PACKET_SIZE))
    print("PACKET_SIZE_DEFAULT_DOPPLER: " + str(PACKET_SIZE_DEFAULT_DOPPLER))

    plt.show(block=False)
    #plt.show()

    timer_start = time.time()
    init()
    print ("it took %fs for init()"%(time.time() - timer_start))

    # The thread will get data from queue when available.
    # Then construct complex number array, put it in datamap and pass it back to main script for plotting.
    plot_thread = threading.Thread(target=update_plot_from_file, args=(fig, ax, func, ground_truth))
    plot_thread.daemon = True
    plot_thread.start()
    #update_plot_from_file(fig, bytevecQueue, func)
    
    
    plt.show(block=True)

# ------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------- #
# ----- Thread: Replay plot ----- #
# ----- Read bytes from log file ----- #
def bytes_from_log(filename, chunksize=PACKET_SIZE-8):
    with open(filename, "rb") as f:
        pattern = ""
        count = 0
        while True:
            c = f.read(1)
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
                
                pattern = ""

# ----- Thread: Replay plot ----- #
# ----- prepare the bytevec ----- #
def init():
    global bytevec
    for b in bytes_from_log(filename, PACKET_SIZE_DEFAULT_DOPPLER - 8):
        bytevec.append(bytes([b]))

    print("len of bytevec: " + str(len(bytevec)))
    print("type of element: " + str(type(bytevec[0])))

# ----- Thread: Replay plot ----- #
# ----- Update the plot from log ----- #
def update_plot_from_file(fig, ax, func, ground_truth):
    
    count = 0
    ending = 5
    while True:
        global frame_count
        global bytevec
        # 
        # When having a complete packet, turn that into datavec and pass it back to update()
        # Stops when the rest of file size is not long enough for another packet
        #
        if frame_count * PACKET_SIZE_DEFAULT_DOPPLER + PAYLOAD_START + PAYLOAD_SIZE <= len(bytevec):
            start = frame_count * PACKET_SIZE_DEFAULT_DOPPLER + PAYLOAD_START
            end = frame_count * PACKET_SIZE_DEFAULT_DOPPLER + PAYLOAD_START + PAYLOAD_SIZE
            collect_data(start,end)

            start = frame_count * PACKET_SIZE_DEFAULT_DOPPLER + PAYLOAD_START + PAYLOAD_SIZE_DEFAULT
            end = start + PAYLOAD_SIZE
            collect_doppler(start,end)
            print("dopplervec length: " + str(len(dopplervec)))

            timer_start = time.time()
            func(datamap)
            
            print ("it took %fs for update_map()"%(time.time() - timer_start))
            print("[update_plot] len of datamap['azimuth']: " + str(len(datamap['azimuth'])))
            frame_count += 1
            print("[update] frame_count: " + str(frame_count))
        else:
            print("[update_plot] That's all. Ending in " + str(ending) + " seconds")
            time.sleep(1)
            ending -= 1
            if ending <= 0:
                os.kill(os.getpid(), signal.SIGINT)
                sys.exit(0)
            continue
            

        try:
            if not flush_test_data:
                fig.canvas.draw_idle()
            count += 1
            ax.set_title('Azimuth-Range FFT Heatmap: ' + str(frame_count) + ' frames', fontsize=16)
            fig.canvas.set_window_title("frame: " + str(frame_count))
            if ground_truth:
                plt.waitforbuttonpress()
            #time.sleep(10000)
            time.sleep(1e-6)
        except:
            print("something fails here")
            break
        

