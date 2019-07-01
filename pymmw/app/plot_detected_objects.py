#
# Copyright (c) 2018, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# CFAR detected objects - 3D plot
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

def update(data):

    if 'objects' not in data: return
        
    items = data['objects']

    for p in items:
        
        x, y, z, d = p['x'], p['y'], p['z'], p['doppler']
        
        xm, ym, zm = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()                    

        if xm[0] <= x <= xm[1] and ym[0] <= y <= ym[1] and zm[0] <= z <= zm[1]:

            val = min(1.0, d / 65536)
            val = 0.1 * np.log(val + 0.0001) + 1                    

            pt = Point((x, y, z), color=(val, 0.0, 1 - val), size=3, marker='.')
            ax.add_artist(pt)
            
            az, el = ax.azim, ax.elev
            
            if abs(az) > 90: xm = max(xm)
            else: xm = min(xm)
            
            if az < 0: ym = max(ym)
            else: ym = min(ym)
            
            if el < 0: zm = max(zm)
            else: zm = min(zm)

            xz = Point((x, ym, z), color=(0.67, 0.67, 0.67), size=1, marker='.')
            ax.add_artist(xz)

            yz = Point((xm, y, z), color=(0.67, 0.67, 0.67), size=1, marker='.')
            ax.add_artist(yz)

            xy = Point((x, y, zm), color=(0.67, 0.67, 0.67), size=1, marker='.')
            ax.add_artist(xy)                    


if __name__ == "__main__":

    if len (sys.argv[1:]) != 1 :
        print('Usage: {} {}'.format(sys.argv[0].split(os.sep)[-1], '<range_maximum>'))
        sys.exit(1)
        
    try:

        range_max = float(sys.argv[1])        
        d = range_max  # int(math.ceil(range_max))

        # ---

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(1, 1, 1, projection='3d')  # rows, cols, idx
        ax.view_init(azim=-45, elev=15)
        
        move_figure(fig, (0 + 45*2, 0 + 45*2))
        
        fig.canvas.set_window_title('...')
                           
        ax.set_title('CFAR Detection'.format(), fontsize=10)
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        
        ax.set_xlim3d((-d / 2, +d / 2))
        ax.set_ylim3d((0, d))
        ax.set_zlim3d((-d / 2, +d / 2))
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        plt.tight_layout(pad=1)
        
        ax.scatter(xs=[], ys=[], zs=[], marker='.', cmap='jet')

        for child in ax.get_children():
            if isinstance(child, art3d.Path3DCollection):
                child.remove()

        from itertools import product, combinations  # a small cube (origin)        
        r = [-0.075, +0.075]
        for s, e in combinations(np.array(list(product(r,r,r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s,e), color="black", linewidth=0.5)

        set_aspect_equal_3d(ax)

        mpl.colors._colors_full_map.cache.clear()  # avoid memory leak by clearing the cache
                    
        start_plot(fig, ax, update)
    
    except Exception:
        sys.exit(2)
