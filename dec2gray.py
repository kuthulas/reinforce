import sys

def bin2gray(bits):
    return bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]

print bin2gray([int(x) for x in bin(int(sys.argv[1]))[2:]])