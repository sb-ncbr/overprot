# Module for creating protein topology diagrams

import svgwrite
import math

################################################################################

def log(*args):
	print(*args)

_EPSILON = 1e-6

def add_arc(drawing, group, p0, p1, semiaxes, stroke='black', fill='none'):
	""" Adds an arc that bulges to the right as it moves from p0 to p1 """
	args = {'x0':p0[0], 
		'y0':p0[1], 
		'xradius':semiaxes[0], 
		'yradius':semiaxes[1] if len(semiaxes) > 1 else semiaxes[0], 
		'ellipseRotation':0, #has no effect for circles
		'x1':(p1[0]-p0[0]), 
		'y1':(p1[1]-p0[1])}
	group.add(drawing.path(d="M %(x0)f,%(y0)f a %(xradius)f,%(yradius)f %(ellipseRotation)f 0,0 %(x1)f,%(y1)f" % args,
			fill=fill, 
			stroke=stroke
			))
def add_bezier_corner(drawing, p0, p1, p2, radius, **kwargs):
	def add(u, v): 
		return (u[0]+v[0], u[1]+v[1])
	def subtract(u, v): 
		return (u[0]-v[0], u[1]-v[1])
	def resize(u, size=1.0):
		ratio = size / math.sqrt(u[0]**2 + u[1]**2)
		return (ratio * u[0], ratio * u[1])
	path = 'M %(x)f,%(y)f' % {'x':p0[0], 'y':p0[1]}
	r01 = resize(subtract(p0,p1), radius)
	r21 = resize(subtract(p2,p1), radius)
	p01 = add(p1, r01)
	p21 = add(p1, r21)
	path += ' L %(x)f,%(y)f' % {'x':p01[0], 'y':p01[1]}
	#path += ' L %(x)f,%(y)f' % {'x':p1[0], 'y':p1[1]}
	#path += ' L %(x)f,%(y)f' % {'x':p21[0], 'y':p21[1]}
	path += ' Q %(x1)f,%(y1)f,%(x)f,%(y)f' % {'x1':p1[0], 'y1':p1[1],'x':p21[0], 'y':p21[1]}
	path += ' L %(x)f,%(y)f' % {'x':p2[0], 'y':p2[1]}
	drawing.add(drawing.path(d=path, **kwargs))

def _add_vectors(u, v): 
	return (u[0]+v[0], u[1]+v[1])
def _subtract_vectors(u, v): 
	return (u[0]-v[0], u[1]-v[1])
def _cross_product_size(u, v):
	return abs(u[0]*v[1] - v[0]*u[1])

def add_smooth_polyline(drawing, group, points, radius, **kwargs):
	n = len(points)
	def center(u, v):
		return (0.5*(u[0]+v[0]), 0.5*(u[1]+v[1]))
	def shrink(u, size):
		old_size = math.sqrt(u[0]**2 + u[1]**2)
		ratio = size / old_size if old_size > size else 1.0  # avoids division by zero
		return (ratio * u[0], ratio * u[1])
	if n <= 1:
		return
	else:
		first = points[0]
		last = points[-1]
		path = 'M %(x)f,%(y)f' % {'x':first[0], 'y':first[1]}
		for i in range(0, n-2):
			p, q, r = points[i:i+3]
			p_ = p if i==0 else center(p, q)
			r_ = r if i==n-3 else center(q, r)
			pq = _add_vectors(q, shrink(_subtract_vectors(p_, q), radius))
			qr = _add_vectors(q, shrink(_subtract_vectors(r_, q), radius))
			path += ' L %(x)f,%(y)f' % {'x':pq[0], 'y':pq[1]}
			path += ' Q %(x1)f,%(y1)f,%(x)f,%(y)f' % {'x1':q[0], 'y1':q[1],'x':qr[0], 'y':qr[1]}
			print('Bezier:',p,pq,q,qr,r)
		path += 'L %(x)f,%(y)f' % {'x':last[0], 'y':last[1]}
		drawing.add(drawing.path(d=path, **kwargs))
	# for point in points:
	# 	drawing.add(drawing.ellipse(center=point, r=(1, 1), **kwargs))
	print(path)

def _flip_vertical(*xs):
	def fv(x):
		if x=='N': 
			return 'S'
		elif x=='S': 
			return 'N'
		elif x == 'E' or x == 'W': 
			return x
		else:
			return (x[0], -x[1])
	return _quasi_map(fv, xs)

def _rotate_left(*xs):
	def rl(x):
		if x=='N': 
			return 'W'
		elif x=='W': 
			return 'S'
		elif x=='S': 
			return 'E'
		elif x=='E': 
			return 'N'
		else:
			return (x[1], -x[0])
	return _quasi_map(rl, xs)

def _rotate_right(*xs):
	def rr(x):
		if x=='N': 
			return 'E'
		elif x=='E': 
			return 'S'
		elif x=='S': 
			return 'W'
		elif x=='W': 
			return 'N'
		else:
			return (-x[1], x[0])
	return _quasi_map(rr, xs)

def _rotate_180(*xs):
	def r180(x):
		if x=='N': 
			return 'S'
		elif x=='S': 
			return 'N'
		elif x=='E': 
			return 'W'
		elif x=='W': 
			return 'E'
		else:
			return (-x[0], -x[1])
	return _quasi_map(r180, xs)

def _quasi_map(function, lst):
	if len(lst) == 1:
		return function(lst[0])
	else:
		return [function(x) for x in lst]

def _add_adapter_length(point, direction, adapter_length):
	x, y = point
	r = adapter_length
	if direction == 'N':
		return (x, y-r)
	elif direction == 'S':
		return (x, y+r)
	elif direction == 'E':
		return (x+r, y)
	elif direction == 'W':
		return (x-r, y)
	else:
		raise Exception('Wrong direction: '+str(direction))

def autopath_with_adapters(point0, direction0, point1, direction1, adapter_length):
	second = _add_adapter_length(point0, direction0, adapter_length)
	prelast = _add_adapter_length(point1, direction1, adapter_length)
	automiddle = autopath(second, direction0, prelast, direction1)
	path = remove_redundant_points([point0, *automiddle, point1])
	# path = [point0, *automiddle, point1]
	return path

def autopath(point0, direction0, point1, direction1):
	args = [point0, direction0, point1, direction1]
	if direction0 == 'E':
		if direction1 =='W':
			x0, y0 = point0
			x1, y1 = point1
			if y0 == y1 or x0 == x1:
				return [point0, point1]
			elif x1 > x0:
				xmid = 0.5 * (x0+x1)
				return [point0, (xmid,y0), (xmid,y1), point1]
			else:
				ymid = 0.5 * (y0+y1)
				return [point0, (x0, ymid), (x1, ymid), point1]
		elif direction1 == 'E':
			x0, y0 = point0
			x1, y1 = point1
			if y0 == y1 or x0 == x1:
				return [point0, point1]
			else:
				xmax = max(x0, x1)
				return [point0, (xmax,y0), (xmax,y1), point1]
		elif direction1 == 'N':
			print('Case E - N', point0, point1)
			x0, y0 = point0
			x1, y1 = point1
			if y0 == y1 or x0 == x1:
				return [point0, point1]
			else:
				if x1 > x0 and y1 > y0:
					print(point0, (x1,y0), point1)
					return [point0, (x1,y0), point1]
				else:
					print(point0, (x0,y1), point1)
					return [point0, (x0,y1), point1]
		elif direction1 == 'S':
			return _flip_vertical(*autopath(*_flip_vertical(*args)))
		else:
			raise Exception('Wrong direction: '+str(direction1))
	elif direction0 == 'N':
		return _rotate_left(*autopath(*_rotate_right(*args)))
	elif direction0 == 'S':
		return _rotate_right(*autopath(*_rotate_left(*args)))
	elif direction0 == 'W':
		return _rotate_180(*autopath(*_rotate_180(*args)))
	else:
		raise Exception('Wrong direction: '+str(direction0))

def remove_redundant_points(path):
	n = len(path)
	if n <= 2:
		return path
	result = [path[0]]
	for i in range(len(path)-2):
		p, q, r = path[i:i+3]
		if not _cross_product_size(_subtract_vectors(p, q), _subtract_vectors(q, r)) <= _EPSILON:  # vectors pq and qr are not colinear
			result.append(q)
	result.append(path[-1])
	return result

class Helix:
	def __init__(self, xs, ys, mirror=False, label=None, start=None, end=None, **kwargs):
		try:
			# Assume horizontal helix
			self.x0, self.x1 = xs
			self.y = ys
			self.horizontal = True
		except TypeError:
			try:
				# Nope, it is vertical helix
				self.x = xs
				self.y0, self.y1 = ys
				self.horizontal = False
			except TypeError:
				# Nope, argument are invalid
				raise Exception('TypeError, should be called Helix((x0, x1), y) or Helix(x, (y0, y1))')
		self.label = label
		self.start = start
		self.end = end
		self.mirror = mirror
		self.kwargs = kwargs


class Strand:
	def __init__(self, xs, ys, label=None, start=None, end=None, **kwargs):
		try:
			# Assume horizontal helix
			self.x0, self.x1 = xs
			self.y = ys
			self.horizontal = True
		except TypeError:
			try:
				# Nope, it is vertical helix
				self.x = xs
				self.y0, self.y1 = ys
				self.horizontal = False
			except TypeError:
				# Nope, argument are invalid
				raise Exception('TypeError, should be called Strand((x0, x1), y) or Strand(x, (y0, y1))')
		self.label = label
		self.start = start
		self.end = end
		self.kwargs = kwargs


class Diagram:
	def __init__(self):
		self.HELIX_HALFWIDTH = 12
		self.HELIX_CAP_SEMIAXIS = 5
		self.STRAND_HEAD_HALFWIDTH = 15
		self.STRAND_TAIL_HALFWIDTH = 10
		self.STRAND_HEAD_LENGTH = 15
		self.LABEL_TEXT_LOWER = 4
		self.BORDER_TEXT_LOWER = 4
		self.BORDER_TEXT_OFFSET_HELIX = 12
		self.BORDER_TEXT_OFFSET_STRAND_HEAD = 15
		self.BORDER_TEXT_OFFSET_STRAND_TAIL = 12
		self.CONNECTOR_CORNER_RADIUS = 10
		self.ADAPTER_LENGTH = 15
		self.STROKE = 'black'
		self.FILL = 'beige'
		self.MARGIN = 10
		self.LABEL_FONT = { 'font_size': 12, 'font-weight': 'bold', 'font_family': 'Sans,Arial', 'text_anchor': 'middle'}  # 'dominant_baseline': 'middle', 'font-weight': 'bold'
		self.BORDER_FONT = { 'font_size': 9, 'font_family': 'Sans,Arial', 'text_anchor': 'middle'}  # 'dominant_baseline': 'middle', 'font_style': 'italic'
		self.drawing = svgwrite.Drawing() 
		self.zero_point = (0, 0)
	
	def _helix_horizontal(self, x_start, x_end, y, stroke=None, fill=None, mirror=False, label=None, start=None, end=None):
		if stroke is None: stroke = self.STROKE
		if fill is None: fill = self.FILL
		dwg = self.drawing
		group = self.drawing
		x_min, x_max = sorted((x_start, x_end))
		x0, x1 = x_min + self.HELIX_CAP_SEMIAXIS, x_max - self.HELIX_CAP_SEMIAXIS
		y0, y1 = y - self.HELIX_HALFWIDTH, y + self.HELIX_HALFWIDTH
		width = abs(x1 - x0)
		height = 2 * self.HELIX_HALFWIDTH
		# Rectangle fill
		group.add(dwg.rect((x0, y0), (width, height), stroke='none', fill=fill))
		# Rectangle stroke
		group.add(dwg.line((x0, y0), (x1, y0), stroke=stroke))
		group.add(dwg.line((x0, y1), (x1, y1), stroke=stroke))
		# Back cap; front cap
		if mirror:
			add_arc(dwg, group, (x0, y0), (x0, y1), (self.HELIX_CAP_SEMIAXIS, self.HELIX_HALFWIDTH), stroke=stroke, fill=fill)
			group.add(dwg.ellipse(center=(x1, y), r=(self.HELIX_CAP_SEMIAXIS, self.HELIX_HALFWIDTH), stroke=stroke, fill=fill))
		else:
			add_arc(dwg, group, (x1, y1), (x1, y0), (self.HELIX_CAP_SEMIAXIS, self.HELIX_HALFWIDTH), stroke=stroke, fill=fill)
			group.add(dwg.ellipse(center=(x0, y), r=(self.HELIX_CAP_SEMIAXIS, self.HELIX_HALFWIDTH), stroke=stroke, fill=fill))
		# Labels
		x_left_text = (x_min if mirror else x_min + 2*self.HELIX_CAP_SEMIAXIS) + self.BORDER_TEXT_OFFSET_HELIX
		x_right_text = (x_max if not mirror else x_max - 2*self.HELIX_CAP_SEMIAXIS) - self.BORDER_TEXT_OFFSET_HELIX
		x_start_text, x_end_text = (x_left_text, x_right_text) if x_start < x_end else (x_right_text, x_left_text)
		x_label = 0.5 * (x_left_text + x_right_text)
		if label is not None:
			group.add(dwg.text(str(label), (x_label, y+self.LABEL_TEXT_LOWER), fill=stroke, **self.LABEL_FONT))
		if start is not None:
			group.add(dwg.text(str(start), (x_start_text, y+self.LABEL_TEXT_LOWER), fill=stroke, **self.BORDER_FONT))
		if end is not None:
			group.add(dwg.text(str(end), (x_end_text, y+self.LABEL_TEXT_LOWER), fill=stroke, **self.BORDER_FONT))
	
	def _helix_vertical(self, y, x_start, x_end, stroke=None, fill=None, mirror=False, label=None, start=None, end=None):
		# In this function x stands for y and vice verse, my refactoring tool is not working, LOL.
		if stroke is None: stroke = self.STROKE
		if fill is None: fill = self.FILL
		dwg = self.drawing
		group = self.drawing
		x_min, x_max = sorted((x_start, x_end))
		x0, x1 = x_min + self.HELIX_CAP_SEMIAXIS, x_max - self.HELIX_CAP_SEMIAXIS
		y0, y1 = y - self.HELIX_HALFWIDTH, y + self.HELIX_HALFWIDTH
		width = abs(x1 - x0)
		height = 2 * self.HELIX_HALFWIDTH
		# Rectangle fill
		group.add(dwg.rect((y0, x0), (height, width), stroke='none', fill=fill))
		# Rectangle stroke
		group.add(dwg.line((y0, x0), (y0, x1), stroke=stroke))
		group.add(dwg.line((y1, x0), (y1, x1), stroke=stroke))
		# Back cap; front cap
		radii = (self.HELIX_HALFWIDTH, self.HELIX_CAP_SEMIAXIS)
		if mirror:
			add_arc(dwg, group, (y1, x0), (y0, x0), radii, stroke=stroke, fill=fill)
			group.add(dwg.ellipse(center=(y, x1), r=radii, stroke=stroke, fill=fill))
		else:
			add_arc(dwg, group, (y0, x1), (y1, x1), radii, stroke=stroke, fill=fill)
			group.add(dwg.ellipse(center=(y, x0), r=radii, stroke=stroke, fill=fill))
		# Labels
		x_left_text = (x_min if mirror else x_min + 2*self.HELIX_CAP_SEMIAXIS) + self.BORDER_TEXT_OFFSET_HELIX
		x_right_text = (x_max if not mirror else x_max - 2*self.HELIX_CAP_SEMIAXIS) - self.BORDER_TEXT_OFFSET_HELIX
		x_start_text, x_end_text = (x_left_text, x_right_text) if x_start < x_end else (x_right_text, x_left_text)
		x_label = 0.5 * (x_left_text + x_right_text)
		if label is not None:
			group.add(dwg.text(str(label), (y, x_label+self.LABEL_TEXT_LOWER), fill=stroke, **self.LABEL_FONT))
		if start is not None:
			group.add(dwg.text(str(start), (y, x_start_text+self.LABEL_TEXT_LOWER), fill=stroke, **self.BORDER_FONT))
		if end is not None:
			group.add(dwg.text(str(end), (y, x_end_text+self.LABEL_TEXT_LOWER), fill=stroke, **self.BORDER_FONT))
	
	def _strand_horizontal(self, x_start, x_end, y, stroke=None, fill=None, label=None, start=None, end=None):
		if stroke is None: stroke = self.STROKE
		if fill is None: fill = self.FILL
		dwg = self.drawing
		group = self.drawing
		x0 = x_start
		x1 = x_end - self.STRAND_HEAD_LENGTH if x_end > x_start else x_end + self.STRAND_HEAD_LENGTH
		x2 = x_end
		y0 = y - self.STRAND_HEAD_HALFWIDTH
		y1 = y - self.STRAND_TAIL_HALFWIDTH
		y2 = y
		y3 = y + self.STRAND_TAIL_HALFWIDTH
		y4 = y + self.STRAND_HEAD_HALFWIDTH
		# Arrow shape
		group.add(dwg.polygon([(x0,y1), (x1,y1), (x1,y0), (x2,y2), (x1,y4), (x1,y3), (x0,y3)], stroke=stroke, fill=fill))
		# Labels
		x_start_text = x_start + self.BORDER_TEXT_OFFSET_STRAND_TAIL if x_start < x_end else x_start - self.BORDER_TEXT_OFFSET_STRAND_TAIL
		x_end_text = x_end - self.BORDER_TEXT_OFFSET_STRAND_HEAD if x_start < x_end else x_end + self.BORDER_TEXT_OFFSET_STRAND_HEAD
		x_label = 0.5 * (x_start_text + x_end_text)
		if label is not None:
			group.add(dwg.text(str(label), (x_label, y+self.LABEL_TEXT_LOWER), fill=stroke, **self.LABEL_FONT))
		if start is not None:
			group.add(dwg.text(str(start), (x_start_text, y+self.LABEL_TEXT_LOWER), fill=stroke, **self.BORDER_FONT))
		if end is not None:
			group.add(dwg.text(str(end), (x_end_text, y+self.LABEL_TEXT_LOWER), fill=stroke, **self.BORDER_FONT))
	
	def _strand_vertical(self, y, x_start, x_end, stroke=None, fill=None, label=None, start=None, end=None):
		# In this function x stands for y and vice verse, my refactoring tool is not working, LOL.
		if stroke is None: stroke = self.STROKE
		if fill is None: fill = self.FILL
		dwg = self.drawing
		group = self.drawing
		x0 = x_start
		x1 = x_end - self.STRAND_HEAD_LENGTH if x_end > x_start else x_end + self.STRAND_HEAD_LENGTH
		x2 = x_end
		y0 = y - self.STRAND_HEAD_HALFWIDTH
		y1 = y - self.STRAND_TAIL_HALFWIDTH
		y2 = y
		y3 = y + self.STRAND_TAIL_HALFWIDTH
		y4 = y + self.STRAND_HEAD_HALFWIDTH
		# Arrow shape
		group.add(dwg.polygon([(y1,x0), (y1,x1), (y0,x1), (y2,x2), (y4,x1), (y3,x1), (y3,x0)], stroke=stroke, fill=fill))
		# Labels
		x_start_text = x_start + self.BORDER_TEXT_OFFSET_STRAND_TAIL if x_start < x_end else x_start - self.BORDER_TEXT_OFFSET_STRAND_TAIL
		x_end_text = x_end - self.BORDER_TEXT_OFFSET_STRAND_HEAD if x_start < x_end else x_end + self.BORDER_TEXT_OFFSET_STRAND_HEAD
		x_label = 0.5 * (x_start_text + x_end_text)
		if label is not None:
			group.add(dwg.text(str(label), (y, x_label+self.LABEL_TEXT_LOWER), fill=stroke, **self.LABEL_FONT))
		if start is not None:
			group.add(dwg.text(str(start), (y, x_start_text+self.LABEL_TEXT_LOWER), fill=stroke, **self.BORDER_FONT))
		if end is not None:
			group.add(dwg.text(str(end), (y, x_end_text+self.LABEL_TEXT_LOWER), fill=stroke, **self.BORDER_FONT))

	def _get_helix_start_point(self, helix):
		if helix.horizontal:
			if not helix.mirror and helix.x0 < helix.x1:
				x = helix.x0 + self.HELIX_CAP_SEMIAXIS
			elif helix.mirror and helix.x0 > helix.x1:
				x = helix.x0 - self.HELIX_CAP_SEMIAXIS
			else:
				x = helix.x0
			return x, helix.y
		else:
			if not helix.mirror and helix.y0 < helix.y1:
				y = helix.y0 + self.HELIX_CAP_SEMIAXIS
			elif helix.mirror and helix.y0 > helix.y1:
				y = helix.y0 - self.HELIX_CAP_SEMIAXIS
			else:
				y = helix.y0
			return helix.x, y

	def _get_helix_end_point(self, helix):
		if helix.horizontal:
			if helix.mirror and helix.x0 < helix.x1:
				x = helix.x1 - self.HELIX_CAP_SEMIAXIS
			elif not helix.mirror and helix.x0 > helix.x1:
				x = helix.x1 + self.HELIX_CAP_SEMIAXIS
			else:
				x = helix.x1
			return x, helix.y
		else:
			if helix.mirror and helix.y0 < helix.y1:
				y = helix.y1 - self.HELIX_CAP_SEMIAXIS
			elif not helix.mirror and helix.y0 > helix.y1:
				y = helix.y1 + self.HELIX_CAP_SEMIAXIS
			else:
				y = helix.y1
			return helix.x, y

	def _get_strand_start_point(self, strand):
		if strand.horizontal:
			return strand.x0, strand.y
		else:
			return strand.x, strand.y0

	def _get_strand_end_point(self, strand):
		if strand.horizontal:
			return strand.x1, strand.y
		else:
			return strand.x, strand.y1

	def get_element_start_point(self, elem):
		if isinstance(elem, Helix):
			return self._get_helix_start_point(elem)
		elif isinstance(elem, Strand):
			return self._get_strand_start_point(elem)
		else:
			raise TypeError('Unknown type of element')
	
	def get_element_end_point(self, elem):
		if isinstance(elem, Helix):
			return self._get_helix_end_point(elem)
		elif isinstance(elem, Strand):
			return self._get_strand_end_point(elem)
		else:
			raise TypeError('Unknown type of element')

	def get_element_start_direction(self, elem):
		return _rotate_180(self.get_element_end_direction(elem))

	def get_element_end_direction(self, elem):
		if elem.horizontal:
			if elem.x0 < elem.x1:
				return 'E'
			else:
				return 'W'
		else:
			if elem.y0 < elem.y1:
				return 'S'
			else:
				return 'N'

	def draw_element(self, elem):
		if isinstance(elem, Helix):
			self.draw_helix(elem)
		elif isinstance(elem, Strand):
			self.draw_strand(elem)
		else:
			raise TypeError('Unknown type of element')

	def draw_helix(self, helix):
		x0, y0 = self.zero_point
		if helix.horizontal:
			self._helix_horizontal(helix.x0 - x0, helix.x1 - x0, helix.y - y0, mirror=helix.mirror, label=helix.label, start=helix.start, end=helix.end, **helix.kwargs)
		else:
			self._helix_vertical(helix.x - x0, helix.y0 - y0, helix.y1 - y0, mirror=helix.mirror, label=helix.label, start=helix.start, end=helix.end, **helix.kwargs)

	def draw_strand(self, strand):
		x0, y0 = self.zero_point
		if strand.horizontal:
			self._strand_horizontal(strand.x0 - x0, strand.x1 - x0, strand.y - y0, label=strand.label, start=strand.start, end=strand.end, **strand.kwargs)
		else:
			self._strand_vertical(strand.x - x0, strand.y0 - y0, strand.y1 - y0, label=strand.label, start=strand.start, end=strand.end, **strand.kwargs)

	def draw_connector(self, points, stroke=None):
		if stroke is None: stroke = self.STROKE
		dwg = self.drawing
		group = self.drawing
		x0, y0 = self.zero_point
		points = [(x - x0, y - y0) for x, y in points]
		print('Draw connector: ', *points)
		#group.add(dwg.polyline(points, stroke=stroke, fill='none'))
		add_smooth_polyline(dwg, dwg, points, self.CONNECTOR_CORNER_RADIUS, stroke=stroke, fill='none')
		
	def saveas(self, filename):
		self.drawing.saveas(filename)

	def get_limits(self, elements):
		xs, ys = [], []
		for elem in elements:
			if isinstance(elem, Helix):
				if elem.horizontal:
					xs.extend((elem.x0, elem.x1))
					ys.extend((elem.y-self.HELIX_HALFWIDTH, elem.y+self.HELIX_HALFWIDTH))
				else:
					xs.extend((elem.x-self.HELIX_HALFWIDTH, elem.x+self.HELIX_HALFWIDTH))
					ys.extend((elem.y0, elem.y1))
			elif isinstance(elem, Strand):
				if elem.horizontal:
					xs.extend((elem.x0, elem.x1))
					ys.extend((elem.y-self.STRAND_HEAD_HALFWIDTH, elem.y+self.STRAND_HEAD_HALFWIDTH))
				else:
					xs.extend((elem.x-self.STRAND_HEAD_HALFWIDTH, elem.x+self.STRAND_HEAD_HALFWIDTH))
					ys.extend((elem.y0, elem.y1))
			elif elem is None:
				# chain interruption
				pass
			else:
				# point
				x, y = elem
				xs.append(x)
				ys.append(y)
		if len(xs) > 0:
			return min(xs), max(xs), min(ys), max(ys)
		else:
			return None

	def add_autopaths(self, elements):
		n = len(elements)
		if n <= 1:
			return elements
		result = [elements[0]]
		for a, b in zip(elements, elements[1:]):
			if isinstance(a, (Helix, Strand)) and isinstance(b, (Helix, Strand)):
				end_a, *path, start_b = autopath_with_adapters(self.get_element_end_point(a), self.get_element_end_direction(a), self.get_element_start_point(b), self.get_element_start_direction(b), self.ADAPTER_LENGTH)
				result.extend(path)
			result.append(b)
		return result
	
	def draw_elements(self, elements, align=True, margin=None, frame_stroke='none', frame_fill='none', autopaths=True):
		if len(elements) == 0:
			return
		if autopaths:
			elements = self.add_autopaths(elements)
		xmin, xmax, ymin, ymax = self.get_limits(elements)
		if margin is None:
			margin = self.MARGIN
		dwg = self.drawing
		if align:
			dwg.add(dwg.rect((0, 0), (xmax-xmin+2*margin, ymax-ymin+2*margin), stroke=frame_stroke, fill=frame_fill))
			self.zero_point = (xmin-margin, ymin-margin)
		else:
			dwg.add(dwg.rect((0, 0), (xmax-xmin+2*margin, ymax-ymin+2*margin), stroke=frame_stroke, fill=frame_fill))
			self.zero_point = (0, 0)
		current_points = []
		last_elem_direction = None
		for elem in elements:
			if isinstance(elem, (Helix, Strand)):
				self.draw_element(elem)
				if len(current_points) > 0:
					current_points.append(self.get_element_start_point(elem))
					self.draw_connector(current_points)
				current_points = [self.get_element_end_point(elem)]
			elif elem is None:
				# chain interruption
				if len(current_points) > 0:
					self.draw_connector(current_points)
					current_points = []
			else:
				# point
				current_points.append(elem)
		self.draw_connector(current_points)


################################################################################

# drawing.saveas(args.output)

dirs = ['N', 'E', (90,50)]
new_dirs = _rotate_left(*dirs)
print(dirs)
print(new_dirs)


d = Diagram()

d.BORDER_FONT['font-style'] = 'italic'
d.BORDER_TEXT_OFFSET_HELIX = 10
d.BORDER_TEXT_OFFSET_STRAND_TAIL = 10

d.draw_elements([
	(-10, 100),
	Helix((0, 100), 100, label='A', start=50, end=61),
	Strand(125, (75, 0), label='1.1', start=65, end=69),
	Strand(160, (0, 75), label='1.2', start=72, end=77),
	Helix((180, 270), 120, label='B', start=80, end=90, mirror=True),
	# Helix(250, (150, 230), label='B', start=105, end=120),
	# Helix(220, (240, 160), label='B', start=105, end=110, mirror=True),
	Strand(195, (0, 75), label='1.3', start=386, end=389),
	# Strand((80, 0), 200, label='1.3', start=105, end=110),
	# Strand((-20, 100), 235, label='1.3', start=105, end=110),
	(195, 85)
], align=True)

d.saveas('diagramos.svg')