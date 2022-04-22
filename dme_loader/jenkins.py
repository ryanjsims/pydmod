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

def hashlittle(key: bytes):
    pass