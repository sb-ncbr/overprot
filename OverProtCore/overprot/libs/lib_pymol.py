'''
Functions utilizing PyMOL.
'''

from __future__ import annotations
import sys
import json
from pathlib import Path
from collections import namedtuple  # change namedtuple to typing.NamedTuple
import numpy as np
from typing import Optional, Sequence, Any, Literal, Iterable

try:
    from pymol import cmd, querying, util, cgo, CmdException  # type: ignore
    cmd.feedback('disable', 'all', 'everything')
    cmd.set('cif_use_auth', 0)
    cmd.set('cif_keepinmemory', 1)
except ImportError:
    print('', file=sys.stderr)

from . import lib
from . import lib_sses
from . import lib_domains
from .lib_logging import ProgressBar
from .lib_structure import Structure
from . import superimpose3d


_MIN_CYLINDER_RADIUS = 0.0  # 0.1  # when radius is dependent on occurrence
_MAX_CYLINDER_RADIUS = 1.6
_DEFAULT_CYLINDER_RADIUS = 0.25

_MIN_ARROW_WIDTH = 0.0  # 0.1  # when radius is dependent on occurrence
_MAX_ARROW_WIDTH = 1.4
_DEFAULT_ARROW_WIDTH = 0.3

_ARROW_HEAD_TAIL_WIDTH_RATIO = 1.6
_ARROW_THICKNESS_WIDTH_RATIO = 0.25
_DARKER_COLOR_RATIO = 0.6


Coloring = Literal['rainbow', 'color']  # coloring scheme in PyMOL sessions, 'rainbow' = blue to red, 'color' = random-like color sequence
CealignResult = namedtuple('CealignResult', ['alignment_length', 'RMSD', 'rotation_matrix', 'rotation', 'translation'])
RotationTranslation = namedtuple('RotationTranslation', ['rotation', 'translation'])


def extract_alpha_trace(input_structfile: Path, output_structfile: Path) -> None:
    obj = 'obj_alpha'
    cmd.load(input_structfile, obj)
    cmd.remove(f'{obj} and not (symbol C and name CA and not hetatm)')
    cmd.save(output_structfile, obj)
    cmd.delete(obj)

def cealign_old(target_file: Path, mobile_file: Path, result_file: Optional[Path] = None) -> CealignResult:
    '''Perform structure superimposition with cealign command and return details.
    If result_file is not None, save transformed mobile structure.'''
    obj_target, obj_mobile = 'obj_cealign_target', 'obj_cealign_mobile'
    cmd.load(target_file, obj_target)
    cmd.load(mobile_file, obj_mobile)
    try:
        if result_file is not None:
            result = cmd.cealign(obj_target, obj_mobile, transform=1)
            cmd.save(result_file, obj_mobile)
        else:
            result = cmd.cealign(obj_target, obj_mobile, transform=0)
    finally:
        cmd.delete(obj_target)
        cmd.delete(obj_mobile)
    result['rotation'], result['translation'] = ttt_matrix_to_rotation_translation(result['rotation_matrix'])
    return CealignResult(**result)
    # TODO optimize by: pre-extracting minimal atom set (cealign uses only alphas by default), only state 1, and center the coordinates
    # TODO optimize by: keeping structures in memory -- probably not needed when working with alpha-traces stored on SSD (2nnj-1tqn: 109ms (with loading) vs 104ms (without loading))

def cealign(target_file: Path, mobile_file: Path, result_file: Optional[Path] = None, fallback_to_dumb_align: bool = False) -> CealignResult:
    '''Perform structure superimposition with cealign command and return details.
    If result_file is not None, save transformed mobile structure.'''
    obj_target, obj_mobile = 'obj_cealign_target', 'obj_cealign_mobile'
    cmd.load(target_file, obj_target)
    cmd.load(mobile_file, obj_mobile)
    try:
        if result_file is not None:
            raw_result = cmd.cealign(obj_target, obj_mobile, transform=1)
            cmd.save(result_file, obj_mobile)
        else:
            raw_result = cmd.cealign(obj_target, obj_mobile, transform=0)

        raw_result['rotation'], raw_result['translation'] = ttt_matrix_to_rotation_translation(raw_result['rotation_matrix'])
        return CealignResult(**raw_result)
    except CmdException:
        target_name = target_file.stem
        mobile_name = mobile_file.stem
        if fallback_to_dumb_align:
            print(f'Warning: cealign({target_name}, {mobile_name}) failed, falling back to dumb_align (internal method)', file=sys.stderr)
            return dumb_align(target_file, mobile_file, result_file)
        else:
            print(f'Warning: cealign({target_name}, {mobile_name}) failed', file=sys.stderr)
            raise
    finally:
        cmd.delete(obj_target)
        cmd.delete(obj_mobile)
    # TODO optimize by: pre-extracting minimal atom set (cealign uses only alphas by default), only state 1, and center the coordinates
    # TODO optimize by: keeping structures in memory -- probably not needed when working with alpha-traces stored on SSD (2nnj-1tqn: 109ms (with loading) vs 104ms (without loading))

def cealign_many(target_file: Path, mobile_files: Sequence[Path], result_files: Sequence[Path], ttt_files: Optional[Sequence[Path]] = None,
                 fallback_to_dumb_align: bool = False, show_progress_bar: bool = False) -> None:
    '''Perform structure superimposition of each mobile to the target with cealign command.
    Save i-th transformed mobile structure into result_files[i].
    Save i-th transformation matrix (PyMOL-style TTT matrix) into ttt_files[i].
    If fallback_to_super, try to use super command if cealign fails.'''
    n = len(mobile_files)
    ttt_files_: Sequence[Path|None]
    if ttt_files is not None:
        ttt_files_ = ttt_files
    else:
        ttt_files_ = [None] * n
    assert len(result_files) == n
    assert len(ttt_files_) == n
    obj_target, obj_mobile = 'obj_cealign_target', 'obj_cealign_mobile'
    with ProgressBar(n, title=f'Cealigning {n} structures', mute = not show_progress_bar) as bar:
        cmd.load(target_file, obj_target)
        for mobile_file, result_file, ttt_file in zip(mobile_files, result_files, ttt_files_):
            cmd.load(mobile_file, obj_mobile)
            try:
                raw_result = cmd.cealign(obj_target, obj_mobile)
                ttt = raw_result['rotation_matrix']
                cmd.save(result_file, obj_mobile)
            except CmdException:
                target_name = target_file.stem
                mobile_name = mobile_file.stem
                if fallback_to_dumb_align:
                    # Cealign is expected to fail on small structures, e.g. CATH family 4.10.180.10
                    print(f'Warning: cealign({target_name}, {mobile_name}) failed, falling back to dumb_align (internal method)', file=sys.stderr)
                    ttt = dumb_align(target_file, mobile_file, result_file).rotation_matrix
                else:
                    print(f'Warning: cealign({target_name}, {mobile_name}) failed', file=sys.stderr)
                    raise
            if ttt_file is not None:
                with open(ttt_file, 'w') as w:
                    print(*ttt, sep='\n', file=w)
            cmd.delete(obj_mobile)
            bar.step()
        cmd.delete(obj_target)

def dumb_align(target_file: Path, mobile_file: Path, result_file: Optional[Path]) -> CealignResult:
    target = read_cif(target_file, only_polymer=True)
    mobile = read_cif(mobile_file, only_polymer=True)
    T = target.coords[:, target.name=='CA']
    M = mobile.coords[:, mobile.name=='CA']
    R, t, rmsd = superimpose3d.dumb_align(M, T)
    ttt = rotation_translation_to_ttt_matrix(RotationTranslation(R, t))
    if result_file is not None:
        mobile.coords = superimpose3d.rotate_and_translate(mobile.coords, R, t)
        with open(result_file, 'w') as w:
            w.write(mobile.to_cif())
    return CealignResult(None, rmsd, ttt, R, t)

def ttt_matrix_to_rotation_translation(flat_ttt_matrix: list[float]) -> RotationTranslation:
    '''Convert PyMOL pseudo-rotation matrix (ttt_matrix) to rotation and translation matrices (A' = rotation @ A + translation)'''
    ttt_matrix = np.array(flat_ttt_matrix).reshape((4,4))
    pretranslation = ttt_matrix[3, 0:3].reshape((3,1))
    rotation = ttt_matrix[0:3, 0:3]
    posttranslation = ttt_matrix[0:3, 3].reshape((3,1))
    translation = rotation @ pretranslation + posttranslation
    return RotationTranslation(rotation, translation)

def rotation_translation_to_ttt_matrix(rot_trans: RotationTranslation) -> list[float]:
    result = np.empty((4, 4), dtype=np.float64)
    result[0:3, 0:3] = rot_trans.rotation
    result[0:3, 3:4] = rot_trans.translation
    result[3, 0:3] = 0.0
    result[3, 3] = 1.0
    return list(result.reshape((16,)))

def read_cif(structfile: Path, only_polymer=False):
    obj = 'obj_read_cif'
    cmd.load(structfile, obj)
    symbol = np.array(querying.cif_get_array(obj, '_atom_site.type_symbol'))
    name = np.array(querying.cif_get_array(obj, '_atom_site.label_atom_id'))
    resn = np.array(querying.cif_get_array(obj, '_atom_site.label_comp_id'))
    resi = np.array(querying.cif_get_array(obj, '_atom_site.label_seq_id'))
    chain = np.array(querying.cif_get_array(obj, '_atom_site.label_asym_id'))
    auth_chain = np.array(querying.cif_get_array(obj, '_atom_site.auth_asym_id'))
    entity = np.array(querying.cif_get_array(obj, '_atom_site.label_entity_id'))
    alt = np.array(querying.cif_get_array(obj, '_atom_site.label_alt_id'))
    # group = np.array(querying.cif_get_array(obj, '_atom_site.group_PDB'))  # TODO is this causing problems in PyMOL 2.4???
    coords = querying.get_coordset(obj).transpose()
    n_models = querying.count_states(obj)
    if n_models > 1:
        print(f'WARNING: {structfile} contains multiple models. Assuming that atoms are sorted by the model ID and taking only first n_atoms atoms', file=sys.stderr)
        n_atoms = coords.shape[-1]
        if symbol.shape[-1] > n_atoms:
            symbol = symbol[0:n_atoms]
            name = name[0:n_atoms]
            resn = resn[0:n_atoms]
            resi = resi[0:n_atoms]
            chain = chain[0:n_atoms]
            auth_chain = auth_chain[0:n_atoms]
            entity = entity[0:n_atoms]
            alt = alt[0:n_atoms]
    cmd.delete(obj)
    result = Structure(symbol=symbol, name=name, resn=resn, resi=resi, chain=chain, auth_chain=auth_chain, entity=entity, coords=coords)
    if only_polymer:
        result = result.filter(result.resi != None)
    return result
    # TODO test on more entries (possible with multiple states), maybe the atom order will be wrong

def create_alignment_session(structfileA: Path, structfileB: Path, alignment_file: Path, output_session_file: Path):
    objA = structfileA.stem
    objB = structfileB.stem
    obj_aln = 'alignment'
    with open(alignment_file) as f:
        alignment = json.load(f)
    cmd.load(structfileA, objA)
    cmd.load(structfileB, objB)

    for chainA, resiA, chainB, resiB in alignment['aligned_residues']:
        atomA = f'{objA} and chain {chainA} and resi {resiA} and name CA'
        atomB = f'{objB} and chain {chainB} and resi {resiB} and name CA'
        cmd.distance(obj_aln, atomA, atomB)
    
    cmd.hide()
    cmd.show('ribbon')
    cmd.show('dashes')
    cmd.show('spheres')
    cmd.set('sphere_scale', 0.2)
    cmd.color('yellow', objA)
    cmd.color('magenta', objB)
    cmd.color('white', obj_aln)
    
    # assert output_session_file.endswith('.pse')
    cmd.save(output_session_file, format='pse')


def create_consensus_session(consensus_structure_file: Path, consensus_sse_file: Path, 
                             out_session_file: Optional[Path], coloring: Coloring = 'rainbow', 
                             out_image_file: Optional[Path] = None, image_size: tuple[int, int] | tuple[int] | tuple[()] = (),
                             enable_occurrence_threshold: float = 0.05) -> None:
    '''image_size is either (width, height) or (width,) (height is scaled) or () (default size)'''
    obj = 'cons'
    group_name = _segment_group_name(obj)
    cmd.load(consensus_structure_file, obj)
    if consensus_sse_file.is_file():
        with open(consensus_sse_file) as f:
            consensus = json.load(f)	
        cmd.group(group_name)
        for sse in consensus['consensus']['secondary_structure_elements']:
            _create_line_segment(sse, group_name, coloring, enable_occurrence_threshold=enable_occurrence_threshold)
    cmd.hide()
    cmd.show('cartoon', obj)
    util.chainbow(obj)  # cmd.color('white', obj)
    cmd.show('cgo', group_name)
    cmd.show('dashes', group_name)
    cmd.set('dash_gap', 0)
    cmd.dss(obj)
    cmd.zoom('vis')
    if out_image_file is not None:
        if image_size is not None:
            cmd.ray(*image_size)
        else:
            cmd.ray()
        cmd.save(out_image_file)
    if out_session_file is not None:
        cmd.save(out_session_file)
        cmd.delete('all')

def create_multi_session(directory: Path, consensus_structure_file: Optional[Path], 
                         consensus_sse_file: Optional[Path], out_session_file: Optional[Path],  
                         coloring: Coloring = 'rainbow', base_color: str = 'gray80', progress_bar: bool = False):
    if consensus_structure_file is not None or consensus_sse_file is not None:
        assert consensus_structure_file is not None, 'consensus_structure_file and consensus_sse_file must be both strings or both None'
        assert consensus_sse_file is not None, 'consensus_structure_file and consensus_sse_file must be both strings or both None'
        create_consensus_session(consensus_structure_file, consensus_sse_file, None, coloring=coloring)
    domains = lib_domains.load_domain_list(directory/'sample.json')
    n = len(domains)
    with ProgressBar((n+1)*n//2, title=f'Creating PyMOL session with {n} structures', mute = not progress_bar) as bar:  # Complexity is ~ quadratic with current PyMOL 2.3
        for i, domain in enumerate(domains):
            cmd.load(directory/f'{domain.name}.cif', domain.name)
            _color_by_annotation(domain.name, directory/f'{domain.name}-clust.sses.json', base_color, coloring, show_line_segments=True)
            bar.step(i)
    cmd.hide()
    cmd.show('cartoon')
    cmd.show('cgo')
    cmd.show('dashes')
    cmd.set('dash_gap', 0)
    cmd.zoom('vis')
    if out_session_file is not None:
        cmd.save(out_session_file)
        cmd.delete('all')

def _color_by_annotation(domain_name: str, annotation_file: Path, base_color: str, coloring: Coloring, show_line_segments: bool = True):
    with open(annotation_file) as f:
        sses = json.load(f)[domain_name]['secondary_structure_elements']
    cmd.color(base_color, f'{domain_name} & symbol C')
    cmd.group(_sses_group_name(domain_name))
    if show_line_segments:
        cmd.group(_segment_group_name(domain_name))
    for sse in sses:
        label = sse['label']
        chain = sse['chain_id']
        start = str(sse['start'])
        end = str(sse['end'])
        color = sse[coloring]
        sel_name = _sses_group_name(domain_name) + '.' + label
        sel_definition = f'{domain_name} & chain {chain} & resi {start}-{end} & symbol C'
        # TODO select only once, then check if empty and possibly delete?
        if cmd.count_atoms(sel_definition)>0:
            cmd.select(sel_name, sel_definition)
            cmd.color(color, sel_name)
        if show_line_segments:
            _create_line_segment(sse, _segment_group_name(domain_name), coloring, enable_occurrence_threshold=0.0)
        cmd.deselect()
    cmd.dss(domain_name)

def _segment_group_name(domain_name: str) -> str:
    return domain_name + '_seg'

def _sses_group_name(domain_name: str) -> str:
    return domain_name + '_sses'

def _create_line_segment(sse: dict[str, Any], group_name: str, coloring: Coloring, enable_occurrence_threshold: float) -> None:
    label = sse['label']
    name = group_name + '.' + label
    start_vector = sse['start_vector']
    end_vector = sse['end_vector']
    color = sse[coloring]
    occurrence = sse.get('occurrence')
    if sse['type'] in 'GHIh':  # helix
        _create_line_segment_cylinder(name, start_vector, end_vector, color, occurrence)
        # _create_line_segment_dash(name, start_vector, end_vector, color, occurrence, round_ends=False)
    else:  # strand
        minor_axis = sse.get('minor_axis')
        _create_line_segment_arrow(name, start_vector, end_vector, minor_axis, color, occurrence)
        # _create_line_segment_cylinder(name, start_vector, end_vector, color, occurrence, round_ends=True)
    if occurrence is not None and occurrence < enable_occurrence_threshold:
        cmd.disable(name)

def _create_line_segment_dash(name: str, start_vector: list[float], end_vector: list[float], color: str, occurrence: float|None, round_ends: bool = False) -> None:
    radius = occurrence * (_MAX_CYLINDER_RADIUS-_MIN_CYLINDER_RADIUS) + _MIN_CYLINDER_RADIUS if occurrence is not None else _DEFAULT_CYLINDER_RADIUS
    cmd.pseudoatom('start', pos=start_vector)
    cmd.pseudoatom('end', pos=end_vector)
    cmd.distance(name, 'start', 'end')
    cmd.color(color, name)
    cmd.set('dash_radius', radius, name)
    if not round_ends:
        cmd.set('dash_round_ends', 0, name)
    cmd.delete('start')
    cmd.delete('end')

def _create_line_segment_cylinder(name: str, start_vector: list[float], end_vector: list[float], color: str, occurrence: float|None) -> None:
    radius = occurrence * (_MAX_CYLINDER_RADIUS-_MIN_CYLINDER_RADIUS) + _MIN_CYLINDER_RADIUS if occurrence is not None else _DEFAULT_CYLINDER_RADIUS
    color_rgb = lib_sses.pymol_spectrum_to_rgb(color)
    cgo_arrow = _cgo_cylinder_from_points(start_vector, end_vector, radius=radius, color=color_rgb)
    cmd.load_cgo(cgo_arrow, name)

def _create_line_segment_arrow(name: str, start_vector: list[float], end_vector: list[float], minor_axis: list[float]|None, color: str, occurrence: float|None) -> None:
    width = occurrence * (_MAX_ARROW_WIDTH-_MIN_ARROW_WIDTH) + _MIN_ARROW_WIDTH if occurrence is not None else _DEFAULT_ARROW_WIDTH
    color_rgb = lib_sses.pymol_spectrum_to_rgb(color)
    cgo_arrow = _cgo_arrow_from_points(np.array(start_vector), np.array(end_vector), minor_axis, width=width, color=color_rgb)
    cmd.load_cgo(cgo_arrow, name)

def _create_void_session(out_session_file: str) -> None:
    cmd.save(out_session_file)


def _darker_rgb(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    return r*_DARKER_COLOR_RATIO, g*_DARKER_COLOR_RATIO, b*_DARKER_COLOR_RATIO

def _cgo_cylinder_from_points(p1: np.ndarray, p2: np.ndarray, radius: float, color: tuple[float, float, float]) -> list[float]:
    return [cgo.CYLINDER, *p1, *p2, radius, *color, *color]

def _cgo_arrow_from_points(p1: np.ndarray, p2: np.ndarray, minor_axis: np.ndarray|None, width: float, color: tuple[float, float, float]) -> list[float]:
    u = p2 - p1
    v = minor_axis if minor_axis is not None else _any_normal(u)
    size_u = np.linalg.norm(u)
    u_normalized = u / size_u
    rotation = np.array((u_normalized, v, np.cross(u_normalized, v))).T
    return _cgo_arrow(*_auto_head(size_u), b=width, bh=width*_ARROW_HEAD_TAIL_WIDTH_RATIO, c=width*_ARROW_THICKNESS_WIDTH_RATIO, origin=p1, rotation=rotation, color=color)

def _cgo_arrow(a=8.0, ah=2.0, b=1.0, bh=1.6, c=0.25, scale=1.0, origin=(0.0, 0.0, 0.0), rotation=None, color: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> list[float]:# appr. tail width
    '''Return a 3D arrow as CGO object, pointing in x direction, in xy plane.
    a = tail length
    ah = head length
    b = tail half-width
    bh = head half-width
    c = half-thickness
    scale = rescales all the above parameters
    origin = starting point (center of the arrow's butt)
    '''
    x, y, z = origin
    a *= scale
    ah *= scale
    b *= scale
    bh *= scale
    c *= scale
    coords = np.array([
        a+ah, 0, c,
        a+ah, 0, -c,
        a, -bh, c,
        a, -bh, -c,
        a, -b, c,
        a, -b, -c,
        0, -b, c,
        0, -b, -c,
        0, +b, c,
        0, +b, -c,
        a, +b, c,
        a, +b, -c,
        a, +bh, c,
        a, +bh, -c,
        a+ah, 0, c,
        a+ah, 0, -c,
    ]).reshape((-1, 3)).T
    if rotation is not None:
        coords = rotation @ coords
    coords += np.array(origin).reshape((3, 1))
    top_face = coords[:, 0::2]
    bottom_face = coords[:, 1::2]
    result = _cgo_from_np(cgo.TRIANGLE_STRIP, _darker_rgb(color), coords)
    result.extend(_cgo_from_np(cgo.TRIANGLE_FAN, color, top_face))
    result.extend(_cgo_from_np(cgo.TRIANGLE_FAN, color, bottom_face))
    return result

def _cgo_from_np(cgo_type: float, cgo_color: tuple[float, float, float], vertices: np.ndarray) -> list[float]:
    result = [cgo.BEGIN, cgo_type, cgo.COLOR, *cgo_color]
    for vertex in vertices.T:
        result.append(cgo.VERTEX)
        result.extend(vertex)
    result.append(cgo.END)
    return result

def _auto_head(total_length: float, max_head=6.0, min_head_ratio=0.5) -> tuple[float, float]:
    '''Return auto-determined tail and head length for an arrow with total_length.'''
    head = (total_length * min_head_ratio * max_head) / (max_head + min_head_ratio * total_length)
    tail = total_length - head
    return (tail, head)

def _any_normal(u: np.ndarray) -> np.ndarray:
    '''Return any vector perpendicular to vector u with size 1.'''
    x = np.array((1.0, 0.0, 0.0))
    y = np.array((0.0, 1.0, 0.0))
    vx = np.cross(u, x)
    vy = np.cross(u, y)
    size_vx = np.linalg.norm(vx)
    size_vy = np.linalg.norm(vy)
    if size_vx > size_vy:
        return vx / size_vx
    else:
        return vy / size_vy

