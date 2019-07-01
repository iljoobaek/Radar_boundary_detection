#
# Copyright (c) 2019, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# abstract capture support
#

import sys, threading, json, queue

# ------------------------------------------------

def update_data(q):
    while True:     
        line = sys.stdin.readline()
        try:
            temp = json.loads(line)
            q.put(temp)
        except:
            pass


def update_log(q, func):    
    while True:
        if not q.empty():
            while q.qsize() > 0: item = q.get()
            func(item)


def start_capture(func):
    
    q = queue.Queue()
    
    threading.Thread(target=update_log, args=(q, func)).start()
     
    tu = threading.Thread(target=update_data, args=(q, ))
    tu.daemon = True
    tu.start()
