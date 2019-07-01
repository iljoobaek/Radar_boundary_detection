import sys, time
import serial

device_name = '/dev/tty.usbmodem000000004'
if __name__ == "__main__":
    with serial.Serial(device_name, 961200, timeout=1) as device:
        count = 0
        while True:
            b = device.read(1)
            intbyte = int.from_bytes(b, "little")
            print('%02x' % intbyte, end=" ", flush=True)
            count += 1
            if count % 64 == 0:
                print("")
            #time.sleep(0.01)