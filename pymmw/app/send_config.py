import sys, time
import serial


cmds = []

#device_name = '/dev/tty.usbmodem000000001'
device_name = '/dev/ttyACM0'
#115200, 921600
def send_to_serial():
    print("sleep 5 second for serial")
    for i in range(0,5):
        print(".", end="", flush=True)
        time.sleep(1)
    print("")

    with serial.Serial(device_name, 115200, timeout=1) as device:
        print(device.name)
        for line in cmds:
            line = line + '\n'
            for i in range(1,30):
                print("=== [send] command: " + line.rstrip() + " " + str(i) + " times")
                device.write(line.encode())
                res_count = 0
                res_string = ""
                for res in device:
                    res_string = res.decode('utf-8', errors='replace')
                    res_string = res_string.rstrip()
                    print(res_string)
                    res_count += 1
                    if res_count == 2:
                        break
                print("=== [send] response_string: " + str(res_string))
                if res_string == "Done" or res_string.find("Init Calibration Status") != -1:
                    print("=== [send] Done received! Next.")
                    break


def read_from_file(filename):
    with open(filename, "r") as f:
        for line in f:
            if(line[0] == '%'):
                continue
            line = line.rstrip()
            cmds.append(line)

config_file = ""
if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Provide path to config file")
        sys.exit(1)

    config_file = sys.argv[1]
    print("open: " + config_file)

    read_from_file(config_file)

    send_to_serial()

