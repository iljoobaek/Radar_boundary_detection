import sys

#filename = "azimuth_only.dat"
filename = "DATA.log"

# ----- Magic Word ----- #
magic_word = "0201040306050807"

# ----- Read bytes from the log file ----- #
def bytes_from_file(filename, chunksize=8248):
    with open(filename, "rb") as f:
        pattern = ""
        count = 0
        while True:
            c = f.read(1)
            if not c:
                break
            if pattern == "" or len(pattern) < 16:
                pattern += c.hex()
            else:
                pattern = pattern[2:16]
                pattern += c.hex()

            if pattern == magic_word:
                print("match: " + pattern)
                for it in range(0,16,2):
                    b = pattern[it] + pattern[it+1]
                    yield int(b)

                chunk = f.read(chunksize)
                for b in chunk:
                    yield b
                
                pattern = ""
            
        # read only the first packet
        """ chunk = f.read(chunksize)
        if chunk:
            for b in chunk:
                yield b """
        # Read all of them
        """ while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break """


# ----- The header of the packet ----- #
header = {
    'VERSION' : 8,
    'PACKETLEN' : 12,
    'PLATFORM' : 16,
    'FRAMENUMBER' : 20,
    'CPUCYCLE' : 24,
    'DETECTOBJ' : 28,
    'NUMTLV' : 32
}

# ----- The header of the TLV ----- #
tlvheader = {
    'TLVTYPE' : 36,
    'TLVLEN' : 40
}

# ----- Some global variables ----- #
PAYLOAD_START = 44
PAYLOAD_SIZE = 8192
PACKET_SIZE = 8256

bytevec = []
frame_count = 0

# ----- Read the packet or packets from file ----- #
for b in bytes_from_file(filename):
    bytevec.append(bytes([b]))

print("len of bytevec: " + str(len(bytevec)))
print("type of element: " + str(type(bytevec[0])))



datavec = []

def process_datavec(start, end):
    count = 0
    first , second = 0 , 0
    for byteindex in range(start,end,2):
        intbyte = bytevec[byteindex] + bytevec[byteindex + 1]
        byteint = int.from_bytes(intbyte, byteorder='little', signed=True)
        if count % 2 == 0:
            first = byteint
        if count % 2 == 1:
            second = byteint
            datavec.append(second)
            datavec.append(first)
        count += 1

    print("len of datavec: " + str(len(datavec)))
    print("count: " + str(count))
    print("first 4 numbers(swapped): %d %d %d %d" % (datavec[0], datavec[1], datavec[2], datavec[3]))

while True:
    if frame_count * PACKET_SIZE + PAYLOAD_START + PAYLOAD_SIZE > len(bytevec):
        break
    start = frame_count * PACKET_SIZE + PAYLOAD_START
    end = frame_count * PACKET_SIZE + PAYLOAD_START + PAYLOAD_SIZE
    process_datavec(start, end)
    datavec.clear()
    frame_count += 1


datamap = {
    'azimuth' : datavec
}
