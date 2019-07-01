#
# Copyright (c) 2019, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

#
# capturing and processing for
# TI IWR1443 ES2.0 EVM running the
# capture demo of SDK 1.1.0.2
#

import os, time, sys, threading

from lib.probe import *
from lib.shell import *
from lib.helper import *
from lib.utility import *

# ------------------------------------------------

_meta_ = {
    'dev': 'xWR14xx',
    'mss': 'Capture Demo',
    'ver': '01.01.00.02',
    'cli': 'CaptureDemo:/>',
    'ccs': os.environ.get('CCS_PATH'),  # full path to CCS
    'dbg': 'XDS110',
    'mem': 0x40000,
    'app': {}
    }

# ------------------------------------------------

try:
    import tiflash
except Exception as e:
    print('exception : mss :', e, file=sys.stderr, flush=True)

# ------------------------------------------------

apps = {}

verbose = False

# ------------------------------------------------

def _read_(data, target=sys.stdout):  # called by read_usr and others
    print(data, end='', file=target, flush=True)
    if all((tag in data for tag in (_meta_['dev'], _meta_['mss'], _meta_['ver']))): return None  # reset detected
    if _meta_['cli'] in data: return True  # cli ready
    return False  # unknown


def _init_(prt, dev, cfg):
    if dev is not None:
        try:
            if 'tiflash' not in sys.modules: return
            l3_size = _meta_['mem']
            ccs = _meta_['ccs']
            con = tiflash.get_connections(ccs)
            con = [c for c in con if _meta_['dbg'] in c]
            if len(con) > 0:
                con = con[0]
                frame_values = cfg['profileCfg']['adcSamples'] * num_rx_antenna(cfg) * chirps_per_frame(cfg) 
                value_size = 2 + 2
                count = cfg['frameCfg']['frames']                
                frame_size = frame_values * value_size
                if count == 0:
                    count = max(1, l3_size // frame_size)
                if frame_size * count > l3_size:
                    raise Exception('frame size ({}) exceeds buffer size ({})'.format(frame_size, l3_size))
                tmem = threading.Thread(
                    target=read_mem,
                    args=(con, dev._serno_, value_size, frame_values, count, prt, cfg['frameCfg']['frames'] == 0))
                tmem.start()
        except Exception as e:
            print('exception : mss :', e, file=sys.stderr, flush=True)


def _conf_(cfg):
    
    global verbose
    
    key = _meta_['dev']

    c = dict(cfg)
    
    if '_comment_' in c:
        c.pop('_comment_', None)  # remove entry        
    
    if '_settings_' in c:
        
        rx_ant = int(c['_settings_']['rxAntennas'])
        tx_ant = int(c['_settings_']['txAntennas'])
        
        # common
        if c['channelCfg']['rxMask'] is None:
            c['channelCfg']['rxMask'] = 2**rx_ant - 1

        if c['channelCfg']['txMask'] is None:
            n = tx_ant
            if n == 1: n = 0
            else: n = 2 * n
            c['channelCfg']['txMask'] = 1 + n
            
        # cli output
        if 'verbose' in c['_settings_'] and c['_settings_']['verbose'] is not None:
            verbose = c['_settings_']['verbose']
                
        c.pop('_settings_', None)  # remove entry
                
    return c, None


def _proc_(cfg, par):
    global apps

    errcode = {None: 'pass', 1: 'miss', 2: 'exec', 3: 'plot'}

    for _, app in apps.items():
        app.kill()
    
    apps.clear()

    for cmd, app in _meta_['app'].items():
        if type(app) not in (list, tuple): app = (app,)             
        for item in app:
            if item not in apps:
                apps[item], values = exec_app(item, (cfg, par, ))
                if values is None: values = []
                print(item, values, ':', errcode[apps[item].poll()], file=sys.stderr, flush=True)


def _pipe_(dat):
    for tag in apps:
        try:
            apps[tag].stdin.write(str.encode(dat + '\n'))
            apps[tag].stdin.flush()
        except:
            pass

# ------------------------------------------------

def read_mem(con, sn, sval, fval, cnt, prt, infinite=True, width=16):    
    if 'tiflash' not in sys.modules: return
    
    active = True
    while active:
        try:
            time.sleep(0.5)
            buf = tiflash.memory_read(
                address=0x51020000,
                num_bytes=sval * fval * cnt,
                ccs=_meta_['ccs'],
                serno=sn,
                connection=con,
                fresh=True)

        except Exception as e:
            print('exception : mss :', e, file=sys.stderr, flush=True)
            break

        if verbose:
            tmp = dec2hex(buf)
            frames = split(tmp, sval * fval * 2)  # two chars
            for frame in frames:
                print('')
                tmp = split(frame, width * sval)
                for line in tmp:
                    print(' '.join(split(line, sval)))

        if infinite:
            send_config(prt, None, None)
            
        active = infinite
