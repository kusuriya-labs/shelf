import exllamav3_ext

if exllamav3_ext.exl3_mul1_multiplier() != 0x00DCD12D:
    raise RuntimeError("EXL3 拡張の multiplier が mul24 ではありません")
