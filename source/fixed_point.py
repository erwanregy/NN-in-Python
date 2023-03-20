INTEGER_WIDTH = 8
FRACTION_WIDTH = 8

def fixed_to_float(fixed: str) -> float:
    floating = int(fixed, 2)
    if floating & (1 << (len(fixed) - 1)):
        floating -= 1 << len(fixed)
    floating /= (1 << FRACTION_WIDTH)
    return floating

def float_to_fixed(floating: float) -> str:
    fixed = int(floating * (1 << FRACTION_WIDTH))
    fixed = format(fixed & ((1 << INTEGER_WIDTH + FRACTION_WIDTH) - 1), f'0{INTEGER_WIDTH + FRACTION_WIDTH}b')
    return fixed

if __name__ == "__main__":
    # INTEGER_WIDTH = 2
    # FRACTION_WIDTH = 2
    
    # for i in range(2 ** (INTEGER_WIDTH + FRACTION_WIDTH)):
    #     binary = bin(i)[2:].zfill(INTEGER_WIDTH + FRACTION_WIDTH)
    #     float_from_fixed = fixed_to_float(binary)
    #     fixed_from_float = float_to_fixed(float_from_fixed)
    #     print(f"binary: {binary} -> float: {float_from_fixed} -> fixed: {fixed_from_float}")
        
    print(floating := fixed_to_float("1111111111110111"))
    print(float_to_fixed(floating))