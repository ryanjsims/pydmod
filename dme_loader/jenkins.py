from typing import Tuple


def oaat(key: bytes):
    i = 0
    hash = 0
    while i < len(key):
        hash += key[i]
        i += 1
        hash &= 0xFFFFFFFF
        hash += hash << 10
        hash &= 0xFFFFFFFF
        hash ^= hash >> 6
    hash += hash << 3
    hash &= 0xFFFFFFFF
    hash ^= hash >> 11
    hash += hash << 15
    hash &= 0xFFFFFFFF
    return hash

def mix(a: int, b: int, c: int) -> Tuple[int, int, int]:
    a &= 0xFFFFFFFF
    b &= 0xFFFFFFFF
    c &= 0xFFFFFFFF

    a -= b
    a -= c
    a ^= c >> 13
    a &= 0xFFFFFFFF

    b -= c
    b -= a
    b ^= a << 8
    b &= 0xFFFFFFFF

    c -= a
    c -= b
    c ^= b >> 13
    c &= 0xFFFFFFFF

    a -= b
    a -= c
    a ^= c >> 12
    a &= 0xFFFFFFFF

    b -= c
    b -= a
    b ^= a << 16
    b &= 0xFFFFFFFF

    c -= a
    c -= b
    c ^= b >> 5
    c &= 0xFFFFFFFF

    a -= b
    a -= c
    a ^= c >> 3
    a &= 0xFFFFFFFF

    b -= c
    b -= a
    b ^= a << 10
    b &= 0xFFFFFFFF

    c -= a
    c -= b
    c ^= b >> 15
    c &= 0xFFFFFFFF

    return a, b, c

def lookup2(key: bytes, initval: int = 0):
    lenpos = len(key)
    length = lenpos
    a = b = 0x9e3779b9
    c = initval
    p = 0

    while lenpos >= 12:
        a += key[p+0] + key[p+1] << 8 + key[p+2] << 16 + key[p+3] << 24
        a &= 0xFFFFFFFF
        b += key[p+4] + key[p+5] << 8 + key[p+6] << 16 + key[p+7] << 24
        b &= 0xFFFFFFFF
        c += key[p+8] + key[p+9] << 8 + key[p+10] << 16 + key[p+11] << 24
        c &= 0xFFFFFFFF
        a, b, c = mix(a, b, c)
        p += 12
        lenpos -= 12
    
    c += length
    if (lenpos >= 11): c += key[p+10]<<24
    if (lenpos >= 10): c += key[p+9]<<16
    if (lenpos >= 9):  c += key[p+8]<<8
    # the first byte of c is reserved for the length
    if (lenpos >= 8):  b += key[p+7]<<24
    if (lenpos >= 7):  b += key[p+6]<<16
    if (lenpos >= 6):  b += key[p+5]<<8
    if (lenpos >= 5):  b += key[p+4]
    if (lenpos >= 4):  a += key[p+3]<<24
    if (lenpos >= 3):  a += key[p+2]<<16
    if (lenpos >= 2):  a += key[p+1]<<8
    if (lenpos >= 1):  a += key[p+0]

    a, b, c = mix(a, b, c)

    return c & 0xFFFFFFFF