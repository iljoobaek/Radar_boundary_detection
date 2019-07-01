#!/bin/sh
'''which' python3 > /dev/null && exec python3 "$0" "$@" || exec python "$0" "$@"
'''

#
# Copyright (c) 2018, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# starting point of pymmw 
#

import os, sys, glob, serial, threading, json

from lib.shell import *
from lib.probe import *

# ------------------------------------------------

def read_con(prt):  # observe control port and call handler when firmware is recognized

    fw = ['.'.join(os.path.splitext(item)[0].split(os.sep)) for item in glob.glob(os.sep.join(('mss', '*.py')))]
    
    def init(data):
        global mss
        if len(data) > 0 and mss is None: 
            for item in fw:            
                mss = __import__(item, fromlist=('',))
                if mss._read_(data, open(os.devnull, "w")) is None:
                    return True
                mss = None
        return False

    cnt = 0
    ext = {}
 
    try:
        
        while True:            
            data = prt.readline().decode('ascii')            
            if init(data):  # firmware identified      
                break
            elif len(data) > 0:
                print(data, end='', flush=True)
            
        reset = False
        while True:
            if mss._read_(data) is not None:
                if reset:  # reset detected
                    cnt += 1
                    file = open('mss/' + os.path.splitext(mss.__file__.split(os.sep)[-1])[0] + '.cfg', 'r')
                    content = load_config(file)
                    cfg = json.loads(content)
                    cfg, par = mss._conf_(cfg)
                    mss._init_(prt, dev, cfg)
                    mss._proc_(cfg, par)
                    send_config(prt, cfg, mss._read_)
                    print_info(cfg)
                reset = False
            else:
                reset = True
            data = prt.readline().decode('ascii')
            
    except Exception as e:        
        print('exception : control :', e, file=sys.stderr, flush=True)
        sys.exit(1)
            

def read_std():   
    for line in sys.stdin:
        if not line.startswith('%'):
            print(line.rstrip(), file=sys.stderr, flush=True)

# ------------------------------------------------

if __name__ == "__main__":
    
    try:

        dev = None
        
        dev = usb_init()
        xds_reset(dev)
        usb_free(dev)

        mss = None

        con = serial.Serial('/dev/ttyACM0', 115200, timeout=0.01)  # Linux
        #con = serial.Serial('COM3', 115200, timeout=0.01)  # Windows
  
        tstd = threading.Thread(target=read_std, args=(), )        
        tstd.start()

        tusr = threading.Thread(target=read_con, args=(con,))
        tusr.start()
        
    except Exception as e:         
        print('exception : main :', e, file=sys.stderr, flush=True)
        sys.exit(1)
