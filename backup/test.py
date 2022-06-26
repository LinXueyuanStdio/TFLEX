"""
@date: 2021/11/28
@description: null
"""


def num_to_list_of_01(num):
    return list("".join(['{:08b}'.format(i) for i in num.to_bytes(length=32, byteorder='little', signed=True)]))


def index_list_of_ones(list_of_01):
    return [i for i, n in enumerate(list_of_01) if n == "1"]


if __name__ == "__main__":
    import struct

    value = 5.1
    ba = bytearray(struct.pack("d", value))
    print([bin(int("0x%02x" % b, 16)) for b in ba])
    print(list(struct.pack("!f", 5.1)))
    print([bin(i) for i in list(struct.pack("!f", 5.1))])
    print([int("0x%02x" % i, 16) for i in bytearray(struct.pack("q", 100))])
    print(list(bin(-100)))
    print('{:032b}'.format(
        0b1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000))
    print(num_to_list_of_01(100))
    print(num_to_list_of_01(1))
    print(num_to_list_of_01(0))
    print(num_to_list_of_01(-1))
    l = num_to_list_of_01(100)
    print(index_list_of_ones(l))
