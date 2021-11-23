from typing import Tuple

def pymol_spectrum_to_rgb(color: str) -> Tuple[float, float, float]:
	'''Convert color representation from PyMOL spectrum to RGB, e.g. 's500' -> (0.0, 1.0, 0.0) (green).'''
	if color.startswith('s') and len(color) == 4 and color[1:].isdigit():
		i = int(color[1:])
	else:
		raise ValueError("PyMOL spectrum color must be in format 'sXYZ' (e.g. 's523'), not '" + color + "'")
	sector = i * 6 // 1000  # The 6 sectors are: mb, bc, cg, gy, yr, rm
	x = i * 6 % 1000 / 1000  # Position within sector [0.0, 1.0)
	if sector == 0:  # magenta-blue
		return (1.0 - x, 0.0, 1.0)
	elif sector == 1:  # blue-cyan
		return (0.0, x, 1.0)
	elif sector == 2:  # cyan-green
		return (0.0, 1.0, 1.0 - x)
	elif sector == 3:  # green-yellow
		return (x, 1.0, 0.0)
	elif sector == 4:  # yellow-red
		return (1.0, 1.0 - x, 0.0)
	elif sector == 5:  # red-magenta
		return (1.0, 0.0, x)

def pymol_spectrum_to_hex(color: str) -> Tuple[float, float, float]:
	'''Convert color representation from PyMOL spectrum to hexadecimal representation, e.g. 's500' -> #00FF00 (green).'''
	r, g, b = pymol_spectrum_to_rgb(color)
	R = int(255*r)
	G = int(255*g)
	B = int(255*b)
	return '#' + hex2(R) + hex2(G) + hex2(B)

def hex2(number: int):
	'''Get two-digit hexadecimal representation of integer from [0, 255], e.g. 10 -> '0A'.'''
	return hex(number)[2:].zfill(2).upper()
