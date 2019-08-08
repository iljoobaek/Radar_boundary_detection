import sys

def read_from_log(filename, length=48):
    with open(filename, "rb") as f:
        return f.read(length)



if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Please provide log path")
        sys.exit(1)
    
    filename = sys.argv[1]
    bytestream = read_from_log(filename)

    print("type" + str(type(bytestream)))
    print("length " + str(len(bytestream)))
    
    print("==== header ====")
    for i, b in enumerate(bytestream):
        print("%02x" % b , end=" ")
        if (i+1) % 16 == 0:
            print("")
    print("================")

    # the document is not correct:
    # we don't have the "sub_frame_number" here
    header = {
        "magic_word": "",
        "version": "",
        "total_packet_length" : 0,
        "platform": "",
        "frame_number": 0,
        "time_cpu_cycle": 0,
        "num_det_obj": 0,
        "num_TLV": 0
    }
    
    
    for i, b in enumerate(bytestream):
        if i < 8 and i % 2 == 0:
            header["magic_word"] += "%02x " % bytestream[i+1]
            header["magic_word"] += "%02x " % bytestream[i]
        
        if i >= 8 and i < 12 and i % 4 == 0:
            header["version"] += "%02x " % bytestream[i+3]
            header["version"] += "%02x " % bytestream[i+2]
            header["version"] += "%02x " % bytestream[i+1]
            header["version"] += "%02x " % bytestream[i]

        if i >= 12 and i < 16 and i % 4 == 0:
            header["total_packet_length"] = int.from_bytes(bytestream[i:i+4], byteorder="little")
        
        if i >= 16 and i < 20 and i % 4 == 0:
            header["platform"] += "%02x " % bytestream[i+3]
            header["platform"] += "%02x " % bytestream[i+2]
            header["platform"] += "%02x " % bytestream[i+1]
            header["platform"] += "%02x " % bytestream[i]
        
        if i >= 20 and i < 24 and i % 4 == 0:
            header["frame_number"] = int.from_bytes(bytestream[i:i+4], byteorder="little")

        if i >= 24 and i < 28 and i % 4 == 0:
            header["time_cpu_cycle"] = int.from_bytes(bytestream[i:i+4], byteorder="little")
        
        if i >= 28 and i < 32 and i % 4 == 0:
            header["num_det_obj"] = int.from_bytes(bytestream[i:i+4], byteorder="little")
        
        if i >= 32 and i < 36 and i % 4 == 0:
            header["num_TLV"] = int.from_bytes(bytestream[i:i+4], byteorder="little")
        
            
            
    
    print("magic_word: " + header["magic_word"])
    print("version: " + header["version"])
    print("total packet length: " + str(header["total_packet_length"]))
    print("platform: " + header["platform"])
    print("frame_number: " + str(header["frame_number"]))
    print("time_cpu_cycle: " + str(header["time_cpu_cycle"]))
    print("num of detected obj: " + str(header["num_det_obj"]))
    print("num of TLV: " + str(header["num_TLV"]))
    #print("sub_frame_number: " + str(header["sub_frame_number"]))



    