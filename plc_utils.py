from snap7.util import *
from snap7.types import *


area = {'I': 0x81, 'Q': 0x82, 'M': 0x83, 'D': 0x84}


def write_data(plc, key, value):
    addr = key.split('.')
    szs = {'x': 1, 'X': 1, 'b': 1, 'B': 1, 'w': 2, 'W': 2, 'd': 4, 'D': 4}

    # Works with Boolean values from a Data Block
    if len(addr) == 3 and addr[0][0] == 'D':
        DBn = int(addr[0][2:])
        DBt = addr[1][2]
        byt = int(addr[1][3:])
        bit = int(addr[2])
        reading = plc.read_area(area['D'], DBn, byt, szs[DBt])
        if DBt == 'X' or DBt == 'x':
            set_bool(reading, 0, bit, value)
        plc.write_area(area['D'], DBn, byt, reading)
    # Works with other data types from a Data Block
    elif len(addr) == 2 and addr[0][0] == 'D':
        DBn = int(addr[0][2:])
        DBt = addr[1][2]
        byt = int(addr[1][3:])
        print(DBn)
        print(byt)
        print(szs[DBt])
        print(plc.get_cpu_state())
        #Found a bug on the line below, Areas.DB was previously ' area['D'] ', saves you some trouble
        reading = plc.read_area(Areas.DB, DBn, byt, szs[DBt])
        print(reading)
        if DBt == 'W' or DBt == 'w':
            set_int(reading, 0, value)
        elif DBt == 'D' or DBt == 'd':
            set_real(reading, 0, value)
        #Areas.DB was previously area['D']
        plc.write_area(Areas.DB, DBn, byt, reading)

    # Works with boolean values from Inputs,Merkels ot Outputs
    elif len(addr) == 2:
        byt = int(addr[0][1:])
        bit = int(addr[1])
        #Found a bug on the line below, Areas.MK was previously ' area[addr[0][0]] '
        reading = plc.read_area(Areas.MK, 0, byt, 1)
        print(Areas.MK)
        set_bool(reading, 0, bit, value)
        plc.write_area(Areas.MK, 0, byt, reading)

    # Works with other data types from Inputs,Merkels ot Outputs eg MW2
    elif len(addr) == 1:
        byt = int(addr[0][2:])
        typ = addr[0][1]
        if typ == 'w' or typ == 'W':
            reading = plc.read_area(area[addr[0][0]], 0, byt, 2)
            set_int(reading, 0, value)
        elif typ == 'd' or typ == 'D':
            reading = plc.read_area(area[addr[0][0]], 0, byt, 4)
            set_real(reading, 0, value)
        plc.write_area(area[addr[0][0]], 0, byt, reading)


def read_data(plc, key, reading=None):
    addr = key.split('.')
    szs = {'x': 1, 'X': 1, 'b': 1, 'B': 1, 'w': 2, 'W': 2, 'd': 4, 'D': 4}

    # Works with Boolean values from a Data Block
    if len(addr) == 3 and addr[0][0] == 'D':
        DBn = int(addr[0][2:])
        DBt = addr[1][2]
        byt = int(addr[1][3:])
        bit = int(addr[2])
        reading = plc.read_area(area['D'], DBn, byt, szs[DBt])
        if DBt == 'X' or DBt == 'x':
            return get_bool(reading, 0, bit)
        else:
            return reading
            # Works with other data types from a Data Block
    elif len(addr) == 2 and addr[0][0] == 'D':
        DBn = int(addr[0][2:])
        DBt = addr[1][2]
        byt = int(addr[1][3:])
        reading = plc.read_area(area['D'], DBn, byt, szs[DBt])
        if DBt == 'W' or DBt == 'w':
            return get_int(reading, 0)
        elif DBt == 'D' or DBt == 'd':
            return get_real(reading, 0)
        #         elif DBt == 'B'or DBt == 'b':
        #             return "".join(map(chr, reading))
        else:
            return reading
    # Works with boolean values from Inputs,Merkels or Outputs
    elif len(addr) == 2:
        byt = int(addr[0][1:])
        bit = int(addr[1])
        reading = plc.read_area(area[addr[0][0]], 0, byt, 1)
        return get_bool(reading, 0, bit)
        # Works with other data types from Inputs,Merkels ot Outputs eg MW2
    elif len(addr) == 1:
        byt = int(addr[0][2:])
        typ = addr[0][1]
        if typ == 'w' or typ == 'W':
            reading = plc.read_area(area[addr[0][0]], 0, byt, 2)
            return get_int(reading, 0)
        elif typ == 'd' or typ == 'D':
            reading = plc.read_area(area[addr[0][0]], 0, byt, 4)
            return get_real(reading, 0)
        else:
            return reading