#
# Copyright (c) 2019, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

import sys, time

from lib.utility import *

# ------------------------------------------------

VID, PID = 0x0451, 0xbef3  # XDS110

# ------------------------------------------------

try:
    import usb
except Exception as e:
    print('exception : lib :', e, file=sys.stderr, flush=True)

# ------------------------------------------------

def usb_init(desc_print=True, nodev_exit=True):
    if 'usb' not in sys.modules: return None    
    try:
        dev = usb.core.find(idVendor=VID, idProduct=PID)
        if dev is not None:
            dev._detached_ = []
            m = usb.util.get_string(dev, dev.iManufacturer)
            p = usb.util.get_string(dev, dev.iProduct)
            s = usb.util.get_string(dev, dev.iSerialNumber)
            dev._serno_ = s
            if desc_print:
                print('{} : {} : {}'.format(m, p, s), file=sys.stderr, flush=True)
            return dev
        elif nodev_exit:
            print('exception : main :', 'no device has been detected', file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(e)
    return None


def usb_point(dev, num, out):
    if 'usb' not in sys.modules: return None    
    ept = (usb.util.ENDPOINT_IN, usb.util.ENDPOINT_OUT)
    cfg = dev.get_active_configuration()        
    intf = cfg[(num, 0)]
    ep = usb.util.find_descriptor(intf,
        custom_match=lambda e: usb.util.endpoint_direction(
            e.bEndpointAddress) == ept[int(out % 2)])
    return ep


def usb_free(dev):
    if 'usb' not in sys.modules: return None
    usb.util.dispose_resources(dev)    
    for ifn in dev._detached_:
        usb.util.release_interface(dev, ifn)
        try: dev.attach_kernel_driver(ifn)
        except: pass

# ------------------------------------------------

def xds_reset(dev, delay=50):
    #_ = {0:'CDC Communication',
    #     1:'CDC Data', 2:'Vendor Specific', 3:'CDC Communication',
    #     4:'CDC Data', 5:'Human Interface Device', 6:'Vendor Specific'}
    ep = usb_point(dev, 2, True)
    if ep is None: return False
    for v in ('00', '01'):
        ep.write(hex2dec('{} {} {} {}'.format('2a', '02', '00', '0e {}'.format(v))))
        time.sleep(delay / 1000)
    return True
