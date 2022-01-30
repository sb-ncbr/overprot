'''
Generate a simple diagram of occurrence and length of SSEs (height and width of the rectangles).

Example usage:
    python3  -m overprot.draw_diagram  input_data/  --dag  --output diagram.svg  --json_output diagram.json
'''

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Optional, Final, Literal, get_args
import svgwrite as svg  # type: ignore

from .libs import lib
from .libs import lib_graphs
from .libs.lib_cli import cli_command, run_cli_command

#  CONSTANTS  ################################################################################

Shape = Literal['rectangle', 'arrow', 'cdf', 'symcdf', 'symcdf0']
SHAPES = get_args(Shape)
SIMPLE_SHAPES = ['rectangle', 'arrow']  # Shapes which do not require CDF
DEFAULT_SHAPE: Final = 'rectangle'

# Aesthetic constants:
X_MARGIN = 5
Y_MARGIN = 5

LENGTH_SCALE = 5.0  # must be float lest svgwrite throw errors
OCCUR_SCALE = 100.0
FLOOR_HEIGHT = 150
FONT_SIZE = 14
UPPER_TEXT_HEIGHT = 20
TEXT_HEIGHT = 20
TEXT_ABOVE_LINE = 5
DEFAULT_TEXT_FILL = 'black'


#  FUNCTIONS  ################################################################################

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

def interpolate(x, ys, xs=None):
    n = len(ys)
    if xs is None:
        xs = np.arange(n) / (n-1)
    if not isinstance(ys, np.ndarray):
        ys = np.array(ys)
    if x < xs[0]:
        return ys[0]
    elif x > xs[n-1]:
        return ys[n-1]
    else:
        for i in range(n-1):
            if xs[i] <= x <= xs[i+1]:
                return ((x - xs[i]) * ys[i+1] + (xs[i+1] - x) * ys[i]) / (xs[i+1] - xs[i])

def heatmap_color(value, norm=50):
    MAX=255
    LOW=50
    x = np.exp(-value / norm)
    colors = [(255, 0, 0), (255, 200, 255), (0, 0, 255)] # from hottest (value inf) to coolest (value 0)
    rgb = interpolate(x, colors)
    return svg.rgb(*rgb)

def renumber_sheets_by_importance(sheet_ids, occurrences, avg_lengths):
    if len(sheet_ids) == 0:
        return sheet_ids.copy()
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

def add_nice_arc(drawing, p0, p1, max_deviation, smart_deviation_param = None, **kwargs):
    if max_deviation == 0:
        drawing.add(drawing.line(p0, p1, **kwargs))
    elif max_deviation < 0:
        add_nice_arc(drawing, p1, p0, -max_deviation, **kwargs)
    else:
        dx = p1[0]-p0[0]
        dy = p1[1]-p0[1]
        sq_distance = dx**2 + dy**2
        if smart_deviation_param is not None:
            distance = np.sqrt(sq_distance)
            max_deviation *= distance / (distance + smart_deviation_param)
        radius = max_deviation / 2 + sq_distance / (8 * max_deviation)
        drawing.add(drawing.path(d=f'M {p1[0]},{p1[1]} a {radius},{radius} 0 0,0 {-dx},{-dy}', **kwargs))

def draw_strand(drawing, left_top, width_height, 
        measure_by_head=False, head_height_ratio=1.4, head_width_fraction=0.5, extra_width_fraction=0, 
        group=None, **kwargs):
    x0_orig, y0_orig = left_top
    width, height = width_height
    augmented_width = width * (1+extra_width_fraction)
    if measure_by_head:
        head_height, tail_height = height, height/head_height_ratio
    else:
        head_height, tail_height = height*head_height_ratio, height
    if group is None:
        group = drawing
    y2 = y0_orig + height/2
    y0 = y2 - head_height/2
    y1 = y2 - tail_height/2
    y3 = y2 + tail_height/2
    y4 = y2 + head_height/2
    x0 = x0_orig - width*extra_width_fraction/2
    x1 = x0 + augmented_width*(1-head_width_fraction)
    x2 = x0 + augmented_width
    group.add(drawing.polygon([(x0,y1), (x1,y1), (x1,y0), (x2,y2), (x1,y4), (x1,y3), (x0,y3)], **kwargs))

def cdf_xy_pairs(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return []
    result = []
    for i, value in enumerate(values):
        if i == 0:
            current_x = value
        elif value > current_x:
            result.append((current_x, i / n))
            current_x = value
    result.append((current_x, 1.0))
    return result

def inverse_cdf_shape(cdf, x0, y0, x_scale, y_scale, include_origin=False, include_zero=False):
    if len(cdf) == 0:
        raise Exception
    first_x, first_y = cdf[0]
    points = []
    if include_origin:
        points.append((0, 0))
    if first_x > 0:
        points.append((0, 1))
    last_y = 0.0
    for x, y in cdf:
        if include_zero or x > 0:
            points.append((x, 1-last_y))
        points.append((x, 1-y))
        last_y = y
    result = [ (x0 + x*x_scale, y0 - y*y_scale) for x, y in points ]
    return result

def symmetrical_inverse_cdf_shape(cdf, x0, y0, x_scale, y_scale, include_zero=False):
    new_x_scale = x_scale / 2
    new_y_scale = y_scale / 2
    quadrant1 = inverse_cdf_shape(cdf, x0, y0, new_x_scale, new_y_scale, include_zero=include_zero)
    quadrant2 = inverse_cdf_shape(cdf, x0, y0, -new_x_scale, new_y_scale, include_zero=include_zero)
    quadrant3 = inverse_cdf_shape(cdf, x0, y0, -new_x_scale, -new_y_scale, include_zero=include_zero)
    quadrant4 = inverse_cdf_shape(cdf, x0, y0, new_x_scale, -new_y_scale, include_zero=include_zero)
    complete = []
    complete.extend(reversed(quadrant1))
    complete.extend(quadrant2)
    complete.extend(reversed(quadrant3))
    complete.extend(quadrant4)
    return complete

def filter_by_occurrence_without_orphan_strands(labels, rel_occurrence, beta_connectivity=[], occurrence_threshold=0):
    '''Return indices of SSEs with rel_occurrence >= occurrence_threshold. Include extra strands with lower occurrence, if necessary in order to avoid orphan strands (i.e. without any neighbour).'''
    n = len(rel_occurrence)
    label_index = { label: i for i, label in enumerate(labels) }
    edges = [ (label_index[l1], label_index[l2]) for l1, l2, *_ in beta_connectivity ]
    neighbours = defaultdict(lambda: set())
    for u, v in edges:
        neighbours[u].add(v)
        neighbours[v].add(u)
    selected = [ occ >= occurrence_threshold for occ in rel_occurrence ]
    for i in range(n):
        if selected[i] and i in neighbours and not any( selected[j] for j in neighbours[i] ):
            mate = max(neighbours[i], key=lambda j: rel_occurrence[j])
            selected[mate] = True
    return [ i for i in range(n) if selected[i] ]

def make_diagram(labels, rel_occurrence, avg_lengths, 
        precedence=None, sheet_ids=None, occurrence_threshold=0, beta_connectivity=[], show_beta_connectivity=False,
        sse_shape='rectangle', cdfs=None, variances=None, label2manual_label={}):  # if precedence is None, the diagram will be drawn as a path, otherwise as a DAG
    indices = filter_by_occurrence_without_orphan_strands(labels, rel_occurrence, beta_connectivity=beta_connectivity, occurrence_threshold=occurrence_threshold)
    
    if len(indices) == 0:
        message = 'No secondary structure elements.'
        dwg = svg.Drawing(style='font-size:' + str(FONT_SIZE)+ ';font-family:Sans, Arial;stroke-width:1;fill:none')
        total_width = FLOOR_HEIGHT + 2*X_MARGIN
        total_height = FLOOR_HEIGHT + 2*Y_MARGIN
        dwg.add(dwg.rect((0, 0), (total_width, total_height), stroke='none', fill='none'))
        dwg.add(dwg.text(message, (X_MARGIN, Y_MARGIN + TEXT_HEIGHT), fill = DEFAULT_TEXT_FILL))
        return dwg

    labels = [labels[i] for i in indices]
    sheet_ids = [sheet_ids[i] for i in indices]
    rel_occurrence = [rel_occurrence[i] for i in indices]
    avg_lengths = [avg_lengths[i] for i in indices]
    if sse_shape not in SIMPLE_SHAPES and cdfs is None:
        raise Exception(f'cdfs cannot be None when sse_shape={sse_shape}')
    if sse_shape in SIMPLE_SHAPES:
        repr_lengths = avg_lengths
    else:
        cdfs = [ cdfs[i] for i in indices ]
        repr_lengths = [ cdf[-1][0] for cdf in cdfs ]
    if precedence is not None:
        precedence = lib.submatrix_int_indexing(precedence, indices, indices)
    if variances is not None:
        variances = [ variances[i] for i in indices ]
    
    n_vertices = len(labels)
    types = [('H' if label[0] in 'GHIh' else 'E') for label in labels]
    sheet_ids = renumber_sheets_by_importance(sheet_ids, rel_occurrence, avg_lengths)

    STRAND2HELIX_WIDTH_RATIO = 2
    avg_lengths = [ length*STRAND2HELIX_WIDTH_RATIO if typ=='E' else length for length, typ in zip(avg_lengths, types) ]

    GAP_RESIDUES = 4
    KNOB_RESIDUES = 1
    
    # Place SSEs into levels (horizontal) and floors (vertical), calculate x starts and ends, edges
    if precedence is not None:  # draw as DAG
        dag = lib_graphs.Dag.from_precedence_matrix(precedence)
    else:  # draw as path
        dag = lib_graphs.Dag.from_path(range(n_vertices))
    vertex_sizes = {i: lib_graphs.Size(repr_lengths[i]*LENGTH_SCALE, rel_occurrence[i]*OCCUR_SCALE) for i in range(n_vertices)}
    box, positions = lib_graphs.embed_dag(dag, vertex_sizes, 
                                          x_padding=GAP_RESIDUES*LENGTH_SCALE, y_padding=FLOOR_HEIGHT-OCCUR_SCALE,
                                          left_margin=X_MARGIN, right_margin=X_MARGIN, 
                                          top_margin=Y_MARGIN+UPPER_TEXT_HEIGHT, bottom_margin=Y_MARGIN+TEXT_HEIGHT,
                                          align_top_left=True)
    total_width = box.right
    total_height = box.bottom
    
    STROKE = svg.rgb(0,0,0)  # color for lines = black
    HELIX_COLOR = svg.rgb(150, 150, 150)  # color for helices = gray
    SHEETS_COLORS = make_color_palette()  # colors for sheets
    fills = { 'H': svg.rgb(255,180,180), 'E': svg.rgb(180,180,255) }
    text_fill = DEFAULT_TEXT_FILL
    dwg = svg.Drawing(style='font-size:' + str(FONT_SIZE)+ ';font-family:Sans, Arial;stroke-width:1;fill:none')
    dwg.add(dwg.rect((0, 0), (total_width, total_height), stroke='none', fill='none'))

    centers_by_label = {}
    fill_colors_by_label = {}

    for u, v in dag.edges:
        x_u, y_u = positions[u]
        x_v, y_v = positions[v]
        x_u_knob = x_u + 0.5*vertex_sizes[u].width + KNOB_RESIDUES*LENGTH_SCALE
        x_v_knob = x_v - 0.5*vertex_sizes[v].width - KNOB_RESIDUES*LENGTH_SCALE
        dwg.add(dwg.line((x_u, y_u), (x_u_knob, y_u), stroke=STROKE))
        dwg.add(dwg.line((x_u_knob, y_u), (x_v_knob, y_v), stroke=STROKE))
        dwg.add(dwg.line((x_v_knob, y_v), (x_v, y_v), stroke=STROKE))

    for i in range(n_vertices):        
        x_middle, y_middle = positions[i]
        width, height = vertex_sizes[i]
        x_left = x_middle - 0.5*width
        y_top = y_middle - 0.5*height
        y_bottom = y_middle + 0.5*height

        y_text = y_bottom + TEXT_HEIGHT
        y_upper_text = y_top - TEXT_ABOVE_LINE
        if variances is not None:
            fill_color = heatmap_color(variances[i]) # debug
        elif sheet_ids is not None:
            sheet_id = int(sheet_ids[i])
            if sheet_id > 0:  # sheet
                fill_color = SHEETS_COLORS[(sheet_id-1) % len(SHEETS_COLORS)]
            else:  # helix
                fill_color = HELIX_COLOR
        else:
            fill_color = fills[types[i]]

        if sse_shape == 'rectangle':
            dwg.add(dwg.rect((x_left, y_top), (width, height), stroke=STROKE, fill=fill_color))
        elif sse_shape == 'arrow':
            is_sheet = sheet_ids is not None and int(sheet_ids[i]) > 0
            if is_sheet:
                draw_strand(dwg, (x_left, y_top), (width, height), stroke=STROKE, fill=fill_color)
            else:
                dwg.add(dwg.rect((x_left, y_top), (width, height), stroke=STROKE, fill=fill_color))
        else:
            if sse_shape == 'cdf':
                polygon = inverse_cdf_shape(cdfs[i], x_left, y_bottom, LENGTH_SCALE, OCCUR_SCALE, include_origin=True)
            elif sse_shape == 'symcdf':
                polygon = symmetrical_inverse_cdf_shape(cdfs[i], x_middle, y_middle, LENGTH_SCALE, OCCUR_SCALE, include_zero=False)
            elif sse_shape == 'symcdf0':
                polygon = symmetrical_inverse_cdf_shape(cdfs[i], x_middle, y_middle, LENGTH_SCALE, OCCUR_SCALE, include_zero=True)
            else:
                raise Exception(f'Unknown sse_shape: {sse_shape}')
            dwg.add(dwg.polygon(polygon, stroke=STROKE, fill=fill_color))
        label = labels[i]
        centers_by_label[label] = (x_middle, y_middle)
        fill_colors_by_label[label] = fill_color
        if label in label2manual_label:
            manual_label = label2manual_label[label]
            dwg.add(dwg.text(manual_label, (x_left, y_upper_text), fill = text_fill))
        dwg.add(dwg.text(label, (x_left, y_text), fill = text_fill))

    if show_beta_connectivity:
        max_deviation = (OCCUR_SCALE/2 + X_MARGIN)
        smart_deviation_param = total_width/5
        for label1, label2, direction in beta_connectivity:
            try:
                add_nice_arc(dwg, centers_by_label[label1], centers_by_label[label2], -direction*max_deviation, smart_deviation_param, stroke=fill_colors_by_label[label1], stroke_width=2)
                add_nice_arc(dwg, centers_by_label[label1], centers_by_label[label2], -direction*max_deviation, smart_deviation_param, stroke=STROKE, stroke_width=0.2)
            except KeyError:
                pass    
    return dwg

def get_label2manual_label(annotation):
    d = {}
    for sse in annotation['secondary_structure_elements']:
        if 'label' in sse and 'manual_label' in sse:
            d[sse['label']] = sse['manual_label']
    return d

def print_json(filename: Path, n_structures, labels, occurrences, avg_lengths, precedence_edges,
        beta_connectivity=None,
        sheet_ids=None,
        cdfs=None,
        variances=None,
        label2manual_label={},
        color_index={}):
    sses = []
    if sheet_ids is not None:
        sheet_ids = renumber_sheets_by_importance(sheet_ids, occurrences, avg_lengths)
    for i, label in enumerate(labels):
        sse = {'label': label, 'occurrence': occurrences[i], 'avg_length': avg_lengths[i]}
        if sheet_ids is not None:
            sse['type'] = 'h' if sheet_ids[i] == 0 else 'e'
            sse['sheet_id'] = sheet_ids[i]
        if label in label2manual_label:
            sse['manual_label'] = label2manual_label[label]
        if variances is not None:
            sse['stdev3d'] = round(variances[i] ** (1/2), ndigits=3)
        if label in color_index:
            sse['rainbow_hex'] = color_index[label]
        if cdfs is not None:
            cdf = [(int(x), float(y)) for (x, y) in cdfs[i]]
            sse['cdf'] = cdf
        sses.append(sse)
    result = { 'n_structures': n_structures, 'nodes': sses, 'precedence': precedence_edges}
    if beta_connectivity is not None:
        label2index = { label: i for i, label in enumerate(labels) }
        result['beta_connectivity'] = [(label2index[l1], label2index[l2], direc) for l1, l2, direc in beta_connectivity]
    lib.dump_json(result, filename, minify=True)

def remove_self_connections(beta_connectivity, print_warnings=False):
    if beta_connectivity is None:
        return None
    else:
        result = [(u, v, *rest) for (u, v, *rest) in beta_connectivity if u != v]
        if print_warnings and len(result) != len(beta_connectivity):
            print(f'WARNING: Beta connectivity contains self-connections.', file=sys.stderr)
        return result


#  MAIN  #####################################################################################

@cli_command()
def main(directory: Path, dag: bool = False, shape: Shape = DEFAULT_SHAPE, connectivity: bool = False, occurrence_threshold: float = 0.0, 
        heatmap: bool = False, output: Optional[Path] = None, json_output: Optional[Path] = None) -> Optional[int]:
    '''Generate a simple diagram of occurrence and length of SSEs (height and width of the rectangles).
    @param  `directory`     Directory with the input files (statistics.tsv, consensus.sses.json, cluster_precedence_matrix.tsv).
    @param  `dag`           Draw the diagram as a DAG instead of a path (requires cluster_precedence_matrix.tsv).
    @param  `shape`         Specify shape of SSEs.
    @param  `connectivity`  Draw arcs to show connectivity of beta-strands (antiparallel connections = up, parallel connections = down).
    @param  `occurrence_threshold`  Do not show SSEs with lower relative occurrence.
    @param  `heatmap`       Colour the SSEs based on their variance of coordinates.
    @param  `output`        Output SVG file (default: <directory>/diagram.svg).
    @param  `json_output`   Output JSON file with preprocessed info.
    '''
    if output is None:
        output = directory/'diagram.svg'

    table, labels, column_names = lib.read_matrix(directory/'statistics.tsv')
    sheet_ids = [ int(x) for x in table[:, column_names.index('sheet_id')] ]
    occurrences = table[:, column_names.index('occurrence')]
    abs_occurrences = table[:, column_names.index('found_in')]
    avg_lengths = table[:, column_names.index('average_length')]
    coord_variances = table[:, column_names.index('coord_variance')]
    n_sses = table.shape[0]
    try:
        n_structures = int(round(max(abs_occurrences) / max(occurrences)))
    except (ZeroDivisionError, ValueError):  # if all SSEs have 0 occurrence or there are 0 SSEs
        n_structures = 0

    precedence: Optional[np.ndarray]
    if dag:
        precedence, *_ = lib.read_matrix(directory/'cluster_precedence_matrix.tsv')
    else:
        precedence = None

    if shape in SIMPLE_SHAPES and json_output is None:
        cdfs = None
    else:
        lengths, _, _ = lib.read_matrix(directory/'lengths.tsv')
        cdfs = [ cdf_xy_pairs(l) for l in lengths ]

    try:
        with open(directory/'consensus.sses.json') as r:
            annot = json.load(r)['consensus']
        beta_connectivity = annot['beta_connectivity']
        label2manual_label = get_label2manual_label(annot)
        color_index = {sse['label']: sse['rainbow_hex'] for sse in annot['secondary_structure_elements']}
    except Exception as ex:
        sys.stderr.write(f'WARNING: Could not read beta-connectivity ({type(ex).__name__}: {ex})\n')
        beta_connectivity = []
        label2manual_label = {}
        color_index = {}
    beta_connectivity = remove_self_connections(beta_connectivity, print_warnings=True)

    if json_output is not None:
        if precedence is not None:
            edges = lib_graphs.Dag.from_precedence_matrix(precedence).edges
        else:
            edges = [(i, i+1) for i in range(len(labels)-1)]
        print_json(json_output, n_structures, labels, occurrences, avg_lengths, edges,
            beta_connectivity=beta_connectivity,
            sheet_ids=sheet_ids,
            cdfs=cdfs,
            variances=coord_variances,
            label2manual_label=label2manual_label,
            color_index=color_index)

    drawing = make_diagram(labels, occurrences, avg_lengths, 
        precedence=precedence,
        sheet_ids=sheet_ids, 
        occurrence_threshold=occurrence_threshold,
        beta_connectivity=beta_connectivity,
        sse_shape=shape, 
        cdfs=cdfs,
        show_beta_connectivity=connectivity,
        variances=coord_variances if heatmap else None,
        label2manual_label=label2manual_label)

    drawing.saveas(output)
    # TODO allow showing manual labels above the rectangles


if __name__ == '__main__':
    run_cli_command(main)
