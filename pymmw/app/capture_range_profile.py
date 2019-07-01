#
# Copyright (c) 2019, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# range and noise profile - capture
#

import os, sys, time

__temp__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__) + '..'))
if __temp__ not in sys.path: sys.path.append(__temp__)

from lib.capture import *

# ------------------------------------------------

def update(data):

    global fh

    x, r, n, o = None, None, None, None
    
    if 'range' in data:
        r = data['range']
        bin = range_max / len(r)
        x = [i*bin for i in range(len(r))]
        x = [v - range_bias for v in x]

    if 'noise' in data:
        n = data['noise']
        if x is None:
            bin = range_max / len(n)
            x = [i*bin for i in range(len(n))]
            x = [v - range_bias for v in x]

    if 'objects' in data and x is not None:
        o = [0] * len(x)
        items = data['objects']
        for p in items:
            ri, _ = p['index']
            o[ri] += 1

    if x is not None:
        clk, cnt = data['header']['time'], data['header']['number']
        if r is None: r = [None] * len(x)
        if n is None: n = [None] * len(x)
        if o is None: o = [0] * len(x)
        for i in range(len(x)):
            s = '{} {:.4f} {:.4f} {:.4f} {}'.format(i, x[i], r[i], n[i], o[i])
            if i == 0: s += ' {} {} {:.3f}'.format(cnt, clk, time.time())
            fh.write(s + '\n')

    fh.flush()  # os.fsync(fh.fileno())
                        
# ------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv[1:]) != 2:
        print('Usage: {} {}'.format(sys.argv[0].split(os.sep)[-1], '<range_maximum> <range_bias>'))
        sys.exit(1)
    
    fh, fp = None, 'log'

    try:

        range_max = float(sys.argv[1])
        range_bias = float(sys.argv[2])
 
        this_name = os.path.basename(sys.argv[0])
        this_name = this_name[len('capture '):-len('.py')]
                 
        if not os.path.exists(fp): os.makedirs(fp)
        fh = open('{}/{}_{}.log'.format(fp, this_name, int(time.time())), 'w')        
        
        start_capture(update)

    except Exception:
        sys.exit(2)
