# Generates a simple diagram of occurrence and length of SSEs (height and width of the rectangles).

import numpy as np
import argparse
import svgwrite as svg
import re
from os import path
import sys
import json
import glob

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('directory', help='Directory with annotation files *-annotated.sses.json', type=str)
parser.add_argument('label', help='SSE label to select', type=str)
args = parser.parse_args()

RE_FILE = re.compile('^.*/(.*)-annotated.sses.json$')

files = glob.glob(args.directory + '/*-annotated.sses.json')
annots = {}
for file in files:
	domain = RE_FILE.sub('\\1', file)
	with open(file) as f:
		annot = json.load(f)
		for v in annot.values():
			annots[domain] = v
			break

for domain, annot in annots.items():
	try:
		the_sse = next( sse for sse in annot['secondary_structure_elements'] if sse['label'] == args.label )
		length = the_sse['end'] - the_sse['start'] + 1
	except StopIteration:
		length = 0
	print(domain, length, sep='\t')

exit()

COMMENT_SYMBOL = '#'
RE_DTYPE = re.compile('^\s*'+re.escape(COMMENT_SYMBOL)+'\s*dtype\s*=\s*(\w+)\s*$')

################################################################################

def log(*args):
	print(*args)

def get_occurrence_and_average(matrix):
	m, n = matrix.shape
	occurence = []
	sums = []
	for i in range(m):
		occurence.append(len([j for j in range(n) if matrix[i, j] !=0]))
		sums.append(sum(matrix[i,:]))
	avg = [sums[i]/occurence[i] for i in range(m)]
	return occurence, avg

def read_matrix(filename, sep='\t', dtype=None):
	with open(filename) as f:
		col_names = None
		row_names = []
		values = []
		for line in iter(f.readline, ''):
			if line.strip() == '':
				# ignore empty line
				pass
			elif line[0] == COMMENT_SYMBOL:
				# read comment
				if dtype==None and RE_DTYPE.match(line):
					dtype = RE_DTYPE.sub('\\1', line)
			elif line[0] == sep:
				# read column names
				_, *col_names = line.strip('\n').split(sep)
				col_names = [name for name in col_names if name != '']
			else:
				# read a row name + values
				row_name, *vals = line.strip('\n').split(sep)
				row_names.append(row_name)
				values.append([float(x) for x in vals if x!=''])
		if col_names==None:
			raise Exception(filename + ' contains no column names')
	matrix = np.array(values, dtype=dtype) if dtype!=None else np.array(values)
	return matrix, row_names, col_names

def make_dag(precedence):
	levels = []
	edges = []
	n = precedence.shape[0]
	# built_precedence = np.zeros(precedence.shape, dtype=bool)  # precedence which is given by transitivity of edges

	def are_transitively_connected(from_level, from_vertex, to_vertex, in_neighbours_of_to_vertex):
		for level in reversed(levels[from_level+1:]):
			for v in level:
				if precedence[from_vertex, v] and v in in_neighbours_of_to_vertex:
					return True
		return False

	todo = set(range(n))
	while len(todo) > 0:
		mins = [i for i in todo if not any(i!=j and precedence[j, i] for j in todo)]
		if len(mins)==0:
			raise Exception('Cyclic graph.')
		for v in mins:
			todo.remove(v)
		for v in mins:
			in_neighbours_of_v = []
			for i in reversed(range(len(levels))):
				for u in levels[i]:
					must_join = precedence[u, v] and not are_transitively_connected(i, u, v, in_neighbours_of_v)
					if must_join:
						edges.append((u, v))
						in_neighbours_of_v.append(u)
		levels.append(mins)
	return levels, edges

def make_color_palette():
	MAX=255
	LOW=150
	LIGHT_COLORS = [
		(MAX, LOW, LOW),  # red
		(LOW, LOW, MAX),  # blue
		(LOW, MAX, LOW),  # green
		(MAX, LOW, MAX),  # magenta
		(LOW, MAX, MAX),  # cyan
		(MAX, MAX, LOW),  # yellow
		]
	MEDIUM_COLORS = [ map(lambda x: x - 80, rgb) for rgb in LIGHT_COLORS ]
	DARK_COLORS = [ map(lambda x: x - 140, rgb) for rgb in LIGHT_COLORS ]
	ALL_COLORS = LIGHT_COLORS + MEDIUM_COLORS + DARK_COLORS
	PALETTE = [svg.rgb(*rgb) for rgb in ALL_COLORS]
	return PALETTE

def renumber_sheets_by_importance(sheet_ids, occurrences, avg_lengths):
	n_sheets = max(sheet_ids) + 1
	importances = [occur * length for occur, length in zip(occurrences, avg_lengths)]
	sheet_importances = [0] * n_sheets
	for i, sheet_id in enumerate(sheet_ids):
		sheet_importances[sheet_id] += importances[i]
	sheets_imps = list(enumerate(sheet_importances))
	sorted_sheets = [0] + [ sheet for sheet, imp in sorted(sheets_imps[1:], key=lambda t: t[1], reverse=True) ]  # put sheet 0 first - it means helices
	ranks = [0] * n_sheets
	for i, sheet in enumerate(sorted_sheets):
		ranks[sheet] = i
	new_ids = [ranks[sheet] for sheet in sheet_ids]
	return new_ids

def submatrix(matrix, row_indices, col_indices):
	m, n = len(row_indices), len(col_indices)
	result = np.zeros((m, n))
	for i in range(m):
		for j in range(n):
			result[i, j] = matrix[row_indices[i], col_indices[j]]
	return result

def make_diagram(labels, rel_occurrence, avg_lengths, precedence=None, sheet_ids=None, occurrence_threshold=0):  # if precedence is None, the diagram will be drawn as a path, otherwise as a DAG
	indices = [i for i, occ in enumerate(rel_occurrence) if occ >= occurrence_threshold]
	labels = [labels[i] for i in indices]
	sheet_ids = [sheet_ids[i] for i in indices]
	rel_occurrence = [rel_occurrence[i] for i in indices]
	avg_lengths = [avg_lengths[i] for i in indices]
	if precedence is not None:
		precedence = submatrix(precedence, indices, indices)
	
	n = len(labels)
	types = [('H' if label[0] in 'GHIh' else 'E') for label in labels]
	sheet_ids = renumber_sheets_by_importance(sheet_ids, rel_occurrence, avg_lengths)

	GAP_RESIDUES = 6
	KNOB_RESIDUES = 1
	
	if precedence is not None:
		# draw as DAG
		levels, edges = make_dag(precedence)
		starts = [0] * n
		ends = [0] * n
		floors = [0] * n
		current_res_count = 0
		max_floors = max(len(level) for level in levels)
		for level in levels:
			for floor, v in enumerate(level):
				starts[v] = current_res_count
				ends[v] = current_res_count + avg_lengths[v]
				floors[v] = floor + (max_floors - len(level)) / 2
			current_res_count = max(ends[v] for v in level) + GAP_RESIDUES
		total_length = current_res_count - GAP_RESIDUES
	else:
		# draw as path
		total_length = sum(avg_lengths) + (n-1)*GAP_RESIDUES
		starts = []
		ends = []
		for i in range(n):
			if len(starts) == 0:
				starts.append(0)
			else:
				starts.append(ends[-1]+GAP_RESIDUES) 
			ends.append(starts[-1]+avg_lengths[i])
		floors = [0] * n
		edges = [(i, i+1) for i in range(n-1)]

	X_MARGIN = 5
	Y_MARGIN = 5

	LENGTH_SCALE = 5
	OCCUR_SCALE = 100
	FLOOR_HEIGHT = 150
	FONT_SIZE = 14
	TEXT_HEIGHT = 20

	max_floor = max(floors)

	Y_CENTER = Y_MARGIN + 0.5 * OCCUR_SCALE
	Y_TEXT = Y_MARGIN + OCCUR_SCALE + TEXT_HEIGHT
	TOTAL_X = 2 * X_MARGIN + LENGTH_SCALE * total_length
	TOTAL_Y = Y_TEXT + Y_MARGIN + FLOOR_HEIGHT * max_floor
	
	stroke = svg.rgb(0,0,0)  # color for lines = gray
	# HELIX_COLOR = svg.rgb(200, 200, 200)  # color for helices = gray
	HELIX_COLOR = svg.rgb(100, 100, 100)  # color for helices = gray
	SHEETS_COLORS = make_color_palette()  # colors for sheets
	fills = { 'H': svg.rgb(255,180,180), 'E': svg.rgb(180,180,255) }
	text_fill = 'black'
	dwg = svg.Drawing(style='font-size:' + str(FONT_SIZE)+ ';font-family:Sans, Arial;stroke-width:1;fill:none')
	dwg.add(dwg.rect((0, 0), (TOTAL_X, TOTAL_Y), stroke='none', fill='none'))

	for i in range(len(starts)):
		x0 = X_MARGIN + LENGTH_SCALE * starts[i]
		y0 = Y_CENTER - 0.5 * OCCUR_SCALE * rel_occurrence[i] + FLOOR_HEIGHT * floors[i]
		width = LENGTH_SCALE * avg_lengths[i] 
		height = OCCUR_SCALE * rel_occurrence[i]
		y_text = Y_CENTER + 0.5 * OCCUR_SCALE * rel_occurrence[i] + TEXT_HEIGHT + FLOOR_HEIGHT * floors[i]
		if sheet_ids is not None:
			sheet_id = int(sheet_ids[i])
			if sheet_id > 0:  # sheet
				fill_color = SHEETS_COLORS[(sheet_id-1) % len(SHEETS_COLORS)]
			else:  # helix
				fill_color = HELIX_COLOR
			# fill_color = SHEETS_COLORS[i % len(SHEETS_COLORS)]			
		else:
			fill_color = fills[types[i]]
		dwg.add(dwg.rect((x0, y0), (width, height), stroke=stroke, fill=fill_color))
		dwg.add(dwg.text(labels[i], (x0, y_text), fill = text_fill))
	for u, v in edges:
		x_u = X_MARGIN + LENGTH_SCALE * ends[u]
		x_v = X_MARGIN + LENGTH_SCALE * starts[v]
		y_u = Y_CENTER + FLOOR_HEIGHT * floors[u]
		y_v = Y_CENTER + FLOOR_HEIGHT * floors[v]
		dwg.add(dwg.line((x_u, y_u), (x_u + KNOB_RESIDUES * LENGTH_SCALE, y_u), stroke=stroke))
		dwg.add(dwg.line((x_u + KNOB_RESIDUES * LENGTH_SCALE, y_u), (x_v - KNOB_RESIDUES * LENGTH_SCALE, y_v), stroke=stroke))
		dwg.add(dwg.line((x_v - KNOB_RESIDUES * LENGTH_SCALE, y_v), (x_v, y_v), stroke=stroke))
	return dwg

################################################################################

# lengths, labels, domains = read_matrix(path.join(args.directory, 'lengths.tsv'))

table, labels, column_names = read_matrix(path.join(args.directory, 'statistics.tsv'))
sheet_ids = [ int(x) for x in table[:, column_names.index('sheet_id')] ]
occurrences = table[:, column_names.index('occurrence')]
avg_lengths = table[:, column_names.index('average_length')]

if args.dag:
	precedence, *_ = read_matrix(path.join(args.directory, 'cluster_precedence_matrix.tsv'))
else:
	precedence = None

drawing = make_diagram(labels, occurrences, avg_lengths, precedence=precedence, sheet_ids=sheet_ids, occurrence_threshold=args.occurrence_threshold)

drawing.saveas(path.join(args.directory, 'diagram.svg'))