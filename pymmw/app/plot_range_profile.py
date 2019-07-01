#
# Copyright (c) 2018, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# range and noise profile - 2D plot
#

import os, sys

try:

    import numpy as np
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    __temp__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__) + '..'))
    if __temp__ not in sys.path: sys.path.append(__temp__)
    
    from lib.plot import *

except ImportError:
    sys.exit(3)

# ------------------------------------------------

def update(data, history=5):

    ax.lines.clear()

    mpl.colors._colors_full_map.cache.clear()

    #for child in ax.get_children():
    #   if isinstance(child, mpl.collections.PathCollection):
    #        child.remove()

    if len(series) > history:
        if series[0] is not None: series[0].remove()
        series.pop(0)

    x = None
    
    if 'range' in data:
        y = data['range']
        bin = range_max / len(y)
        x = [i*bin for i in range(len(y))]
        x = [v - range_bias for v in x]
        ax.plot(x, y, color='blue', linewidth=0.75)

        if 'objects' in data:        
            a = {}
            items = data['objects']
            for p in items:
                ri, di = p['index']
                if ri not in a: a[ri] = {'x': x[ri], 'y': y[ri], 's': 5}
                a[ri]['s'] += 2
            xo = [a[k]['x'] for k in a]
            yo = [a[k]['y'] for k in a]
            so = [a[k]['s'] for k in a]
            path = ax.scatter(xo, yo, c='red', s=so, alpha=0.5)
            series.append(path)
        
        else:
            series.append(None)

    if 'noise' in data:
        y = data['noise']
        if x is None:
            bin = range_max / len(y)
            x = [i*bin for i in range(len(y))]
            x = [v - range_bias for v in x]
        ax.plot(x, y, color='green', linewidth=0.5)
                        
# ------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv[1:]) != 2:
        print('Usage: {} {}'.format(sys.argv[0].split(os.sep)[-1], '<range_maximum> <range_bias>'))
        sys.exit(1)
        
    try:

        range_max = float(sys.argv[1])
        range_bias = float(sys.argv[2])

        # ---

        series = []

        if range_max < 100:  # short range
            fig = plt.figure(figsize=(6, 6))
        else:  # long range
            fig = plt.figure(figsize=(12, 6))
            
        ax = plt.subplot(1, 1, 1)  # rows, cols, idx
        
        move_figure(fig, (0 + 45*1, 0 + 45*1))
        
        fig.canvas.set_window_title('...')
                           
        ax.set_title('Range Profile'.format(), fontsize=10)
        
        ax.set_xlabel('Distance [m]')
        ax.set_ylabel('Relative power [dB]')
        
        ax.set_xlim([0, range_max])
        
        if int(range_max * 100) in (2250, 4093):  # high acc lab
            ax.set_ylim([-5, 105])
            ax.set_yticks(range(0, 100 + 1, 10))             
        
        elif int(range_max * 100) > 10000:  # 4k ffk
            ax.set_ylim([0, 160])
            ax.set_yticks(range(0, 160 + 1, 10))
            ax.set_xticks(range(0, int(range_max) + 5, 10))
            
        else:  # standard
            ax.set_ylim([0, 120])
            ax.set_yticks(range(0, 120 + 1, 10))

        plt.tight_layout(pad=2)
        
        ax.plot(x=[], y=[])
        ax.grid()
        
        start_plot(fig, ax, update)
    
    except Exception:
        sys.exit(2)
