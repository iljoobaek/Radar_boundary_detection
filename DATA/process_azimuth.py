
# ----- Read bytes from the log file ----- #
def bytes_from_file(filename, chunksize=8256):
    with open(filename, "rb") as f:
        # read only the first packet
        """ chunk = f.read(chunksize)
        if chunk:
            for b in chunk:
                yield b """
        # Read all of them
        while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break

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
for b in bytes_from_file('azimuth_only.dat'):
    bytevec.append(bytes([b]))

print("len of bytevec: " + str(len(bytevec)))
print("type of element: " + str(type(bytevec[0])))
'''
# ----- iterate through all the fields in packet header ----- #
for key in header:
    print("field: " + key)
    byteindex = header[key]
    print("bytes from " + str(byteindex) + " to " + str(byteindex + 4))
    for i in range(byteindex , byteindex + 4):
        print(bytevec[i].hex(), end = " ")
    print()

    intbyte = bytevec[byteindex] + bytevec[byteindex + 1] + bytevec[byteindex + 2] + bytevec[byteindex + 3]
    byteint = int.from_bytes(intbyte, "little")
    print("byteint: " + str(byteint))
    print()


# ----- iterate through all the fields in TLV header ----- #
for key in tlvheader:
    print("field: " + key)
    byteindex = tlvheader[key]
    print("bytes from " + str(byteindex) + " to " + str(byteindex + 4))
    for i in range(byteindex , byteindex + 4):
        print(bytevec[i].hex(), end = " ")
    print()

    intbyte = bytevec[byteindex] + bytevec[byteindex + 1] + bytevec[byteindex + 2] + bytevec[byteindex + 3]
    byteint = int.from_bytes(intbyte, "little")
    print("byteint: " + str(byteint))
    print()

'''
# ----- generate the array of Img, Real, Img, Real of each phasor --- #
# 
# The way to do it is to iterate through the 8192 bytes and combine 2 bytes 
# into an integer, something like I did here: int.from_bytes(data, "little")
#
# Maybe I shall think about rearrange the array in Real, Img, Real, Img...
# 
'''
datavec = []
count = 0
for byteindex in range(44,8236,2):
    intbyte = bytevec[byteindex] + bytevec[byteindex + 1]
    byteint = int.from_bytes(intbyte, byteorder='little', signed=True)
    datavec.append(byteint)
    count += 1

print("len of data: " + str(len(datavec)))
print("count: " + str(count))
print("first 4 numbers: %d %d %d %d" % (datavec[0], datavec[1], datavec[2], datavec[3]))
'''

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