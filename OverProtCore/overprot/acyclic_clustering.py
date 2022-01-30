'''
Performs clustering of SSEs from a set of domains, preserving SSE order and type.
Example usage:
    python3  -m overprot.acyclic_clustering  --help
'''

from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Optional, Literal, Dict, Any, List

from .libs import lib
from .libs import lib_sh
from .libs import lib_graphs
from .libs import lib_domains
from .libs import lib_sses
from .libs import lib_acyclic_clustering_simple
from .libs.lib_domains import Domain
from .libs.lib_cli import cli_command, run_cli_command

#  CONSTANTS  ################################################################################

CHAIN_ID_IN_CONSENSUS = 'A'
START_IN_CONSENSUS = 0
END_IN_CONSENSUS = 0


#  FUNCTIONS  ################################################################################

def write_clustered_sses(directory: Path, domain_names: List[str], sse_table, precedence_matrix=None, edges=None):
    sizes, means, variances, covariances, minor_axes = sse_coords_stats(sse_table)
    rainbow_colors = lib_sses.spectrum_colors_weighted(sizes)

    lengths = np.zeros_like(sse_table, dtype=np.int32)
    m, n = sse_table.shape
    types = ['X'] * n
    for j in range(n):
        try:
            types[j] = next( lib_sses.two_class_type(sse_table[i,j]) for i in range(m) if sse_table[i,j] is not None )
        except StopIteration:
            pass
    new_labels = [ typ + str(j) for j, typ in enumerate(types) ]
    for i, domain in enumerate(domain_names):
        these_sses = []
        for j in range(n):
            if sse_table[i,j] is not None:
                sse = dict(sse_table[i,j])  # copy (because of duplicates)
                lengths[i,j] = lib_sses.length(sse)
                sse['label'] = new_labels[j]
                sse['color'] = lib_sses.hash_color(j)
                sse['color_hex'] = lib_sses.pymol_spectrum_to_hex(sse['color'])
                sse['rainbow'] = rainbow_colors[j]
                sse['rainbow_hex'] = lib_sses.pymol_spectrum_to_hex(sse['rainbow'])
                these_sses.append(sse)
        output_json = { domain: { 'secondary_structure_elements': these_sses } }
        lib.dump_json(output_json, directory / f'{domain}-clust.sses.json')
    lib.print_matrix(lengths.transpose(), directory/'lengths.tsv', row_names=new_labels, col_names=domain_names)
    if precedence_matrix is not None:
        lib.print_matrix(precedence_matrix, directory/'cluster_precedence_matrix.tsv', row_names=new_labels, col_names=new_labels)

    # Write consensus
    consensus_sses = []
    for j in range(n):
        sse = {}
        sse['label'] = new_labels[j]
        sse['type'] = types[j]
        sse['chain_id'] = CHAIN_ID_IN_CONSENSUS
        sse['start'] = START_IN_CONSENSUS
        sse['end'] = END_IN_CONSENSUS
        sse['color'] = lib_sses.hash_color(j)
        sse['color_hex'] = lib_sses.pymol_spectrum_to_hex(sse['color'])
        sse['rainbow'] = rainbow_colors[j]
        sse['rainbow_hex'] = lib_sses.pymol_spectrum_to_hex(sse['rainbow'])
        sse['found_in'] = sizes[j]
        sse['occurrence'] = sizes[j] / m
        sse['start_vector'] = means[j][0:3].tolist()
        sse['end_vector'] = means[j][3:6].tolist()
        sse['variance'] = variances[j]
        sse['covariance'] = covariances[j].tolist()
        sse['minor_axis'] = minor_axes[j].tolist() if minor_axes[j] is not None else None
        consensus_sses.append(sse)
    consensus: Dict[str, Any] = {}
    consensus['n_sample'] = m
    consensus['secondary_structure_elements'] = consensus_sses
    if edges is not None:
        beta_connectivity = [ (new_labels[s1], new_labels[s2], typ) for s1, s2, typ in edges ]
        sheets = lib_graphs.connected_components(None, edges)
        for i_sheet, sheet in enumerate(sheets):
            for i_strand in sheet:
                consensus_sses[i_strand]['sheet_id'] = i_sheet + 1
        consensus['beta_connectivity'] = beta_connectivity
    else:
        for sse in consensus_sses:
            if sse is not None and sse['label'].startswith('E'):
                sse['sheet_id'] = 1
    lib.dump_json({'consensus': consensus}, directory/'consensus.sses.json')

    # Write statistics
    occurences = [ size / m for size in sizes ]
    average_lengths = [ lib.safe_mean([x for x in column if x > 0]) for column in lengths.transpose() ]  # type: ignore
    # average_lengths = [ (l if not np.isnan(l) else 0) for l in average_lengths ]
    sheet_ids = [ sse['sheet_id'] if 'sheet_id' in sse else 0 for sse in consensus_sses ]
    stat_table = np.array([sheet_ids, sizes, occurences, average_lengths, variances]).transpose()
    lib.print_matrix(stat_table, directory/'statistics.tsv', row_names=new_labels, col_names=['sheet_id', 'found_in', 'occurrence', 'average_length', 'coord_variance'])

    # Correlation of occurrence
    occ = np.zeros_like(lengths, dtype=np.float64)
    for i in range(m):
        for j in range(n):
            occ[i, j] = 1 if lengths[i, j] != 0 else 0
    # print(occ.transpose())
    # corr = np.nan_to_num(np.corrcoef(occ.transpose()))
    corr = lib.safe_corrcoef(occ.transpose())
    lib.print_matrix(corr, directory/'occurrence_correlation.tsv', row_names=new_labels, col_names=new_labels)
    return

def table_by_pdb_and_label(offsets, n_labels, labels):  # Puts sse indices into a table of lists by their PDB (given by offsets) and label
    n_pdbs = len(offsets) - 1
    table = np.zeros((n_pdbs, n_labels), dtype=list)
    for i in range(n_pdbs):
        for j in range(n_labels):
            table[i, j] = []
    for i_pdb in range(n_pdbs):
        for i_sse in range(offsets[i_pdb], offsets[i_pdb+1]):
            label = labels[i_sse]
            if label >= 0:  # assume there can be unlabeled SSEs (label=-1)
                table[i_pdb, label].append(i_sse)
    return table

def make_hybrid_sses_from_table(table, sses):
    n_pdbs, n_labels = table.shape
    hybrids = np.zeros((n_pdbs, n_labels), dtype=list)
    for i in range(n_pdbs):
        for j in range(n_labels):
            if len(table[i,j]) > 1:
                lib.log(i, j, len(table[i, j]))
            sses_here = lib.unique( sses[idx] for idx in table[i,j] )
            if len(sses_here) == 0:
                hybrids[i,j] = None
            elif len(sses_here) == 1:
                hybrids[i,j] = sses_here[0]
            else:
                # This piece of code was never tested (no data)
                first = min(sses_here, key = lambda s: s['start'])
                last = max(sses_here, key = lambda s: s['end'])
                sse = { 'label': None,
                    'chain_id': first['chain_id'],
                    'start': first['start'],
                    'end': last['end'],
                    'type': lib_sses.two_class_type(first),
                    'start_vector': first['start_vector'],
                    'end_vector': last['end_vector'],
                    'joined_from': sses_here }
                hybrids[i,j] = sse
                raise Exception(f'Yomummafat {i}, {j}.')
    return hybrids

def sse_coords_stats(hybrids):
    n_domains = len(hybrids)
    n_sses = len(hybrids[0])
    sizes, means, variances, covariances, minor_axes = [], [], [], [], []
    for i in range(n_sses):
        sses = [ hybrids[j][i] for j in range(n_domains) if hybrids[j][i] is not None ]
        coords = np.zeros((6, len(sses)), dtype=float)
        for j, sse in enumerate(sses):
            coords[0:3, j] = sse['start_vector']
            coords[3:6, j] = sse['end_vector']
        mean = lib.safe_mean(coords, axis=1)
        minor_axis = combine_strand_minor_axes(sses, mean[3:6] - mean[0:3])
        cov = np.cov(coords) if len(sses) > 1 else np.zeros((6,6), dtype=float)
        var = np.trace(cov)
        sizes.append(len(sses))
        means.append(mean)
        variances.append(var)
        covariances.append(cov)
        minor_axes.append(minor_axis)
    return sizes, means, variances, covariances, minor_axes

def combine_strand_minor_axes(sses: list[dict], major_axis: np.ndarray) -> np.ndarray|None:
    minor_axes = []
    weights = []
    for j, sse in enumerate(sses):
        min_ax = sse.get('minor_axis')
        if min_ax is not None:
            minor_axes.append(min_ax)
            sse_length = sse['end'] - sse['start'] + 1
            weights.append(sse_length)
    n = len(minor_axes)
    if n == 0:
        return None
    minor_axes = np.array(minor_axes).T
    weights = np.array(weights)
    if n == 1:
        result = minor_axes[:, 0]
    elif n == 2:
        u = minor_axes[:, 0]
        v = minor_axes[:, 1]
        if np.dot(u, v) >= 0:
            result = weights[0] * u + weights[1] * v
        else:
            result = weights[0] * u - weights[1] * v
    else:
        M = minor_axes * np.sqrt(weights).reshape((1, -1))
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        result = U[:, 0]
    result = np.cross(np.cross(major_axis, result), major_axis)
    result /= np.linalg.norm(result)
    return result

def calculate_duplicity(n_duplicates, n_1, n_2):
    duplicity = n_duplicates / min(n_1, n_2)
    return duplicity

def guided_clustering_score_function(coords1, weights1, coords2, weights2, 
        shape: Literal['ramp', 'smoothramp', 'exp'] = 'smoothramp',
        d0: float = 30.0, smoothness: float = 0.01, min_score: float = 0.01):
    '''coords1.shape == (n1, d), coords2.shape == (n2, d), weights1.shape == (n1,), weights2.shape == (n2,), result.shape == (n1, n2)'''
    d = 7 # first value is SSE type, values 1:7 are SSE start and end in 3D
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    assert coords1.shape == (n1, d), f'coords1.shape = {coords1.shape}'
    assert coords2.shape == (n2, d), f'coords2.shape = {coords2.shape}'
    assert weights1.shape == (n1,), f'weights1.shape = {weights1.shape}'
    assert weights2.shape == (n2,), f'weights2.shape = {weights2.shape}'
    types1 = coords1[:, 0]
    types2 = coords2[:, 0]
    scores = np.zeros((n1, n2))
    for sse_type in lib_sses.SseType:
        mask1 = (types1 == sse_type)
        mask2 = (types2 == sse_type)
        dist = lib_acyclic_clustering_simple.sse_distance_matrix(coords1[mask1, 1:7], coords2[mask2, 1:7])
        if shape == 'ramp':
            scor = np.maximum((d0 - dist) / d0, min_score)  # type: ignore
        elif shape == 'smoothramp':
            scor = smoothed_ramp(dist, d0, smoothness)
        elif shape == 'exp':
            scor = np.exp(-dist / d0)
        else:
            raise ValueError('shape')
        lib.submatrix_bool_indexing(scores, mask1, mask2, put=scor)
    return scores

def smoothed_ramp(x: np.ndarray, x0: float, a: float) -> np.ndarray:
    '''Smooth and strictly decreasing approximation of function f(x) = max(1-x/k, 0).
    [0, inf) -> (0, 1]
    a is a smoothness parameter from interval [0, 1).
    For a==0, the function is equal to f(x) = max(1-x/x0, 0)'''
    assert 0 <= a < 1
    A = x0 * (1 - a)
    B = x + x0 * (2*a - 1)
    C = -x0 * a
    D = B**2 - 4*A*C
    y1 = (-B + np.sqrt(D)) / (2 * A)
    # y2 = (-B - np.sqrt(D)) / (2 * A)
    return y1

def guided_clustering_sample_aggregation_function(coords1, weight1, coords2, weight2):
    weight = weight1 + weight2
    coords = (coords1 * weight1 + coords2 * weight2) / weight
    return coords, weight

def run_guided_clustering(domains: List[Domain], directory: Path, secstrannotator_rematching: bool = False, fallback: Optional[float] = 0) -> None:
    domain_names = [domain.name for domain in domains]
    offsets, sses, coordinates, type_vector, edges = lib_acyclic_clustering_simple.read_sses_simple(domains, directory)
    n_structures = len(offsets) - 1
    n_sses = len(sses)
    segment_lengths = lib_acyclic_clustering_simple.segment_lengths(coordinates)
    if 0 in segment_lengths:
        raise Exception('Some of the SSE segment lengths are zero. Use newer version of secondary structure assignment with non-zero lengths.')

    # Run clustering
    extended_coords = np.empty((n_sses, 7))
    extended_coords[:, 0] = type_vector
    extended_coords[:, 1:7] = coordinates

    guide_tree_children, _, _ = lib.read_matrix(directory/'guide_tree.children.tsv')

    gac = lib_acyclic_clustering_simple.GuidedAcyclicClustering()
    gac.fit(extended_coords, guided_clustering_score_function, guided_clustering_sample_aggregation_function, offsets, guide_tree_children,
        beta_connections=edges, ladder_correction=True)
    
    table = table_by_pdb_and_label(offsets, gac.n_clusters, gac.labels)
    hybrids = make_hybrid_sses_from_table(table, sses)
    cluster_edges = lib_acyclic_clustering_simple.cluster_edges(edges, gac.labels)
    if any(u == v for u, v, *rest in cluster_edges):
        raise Exception(f'Resulting beta connectivity contains self-connections.')
    write_clustered_sses(directory, domain_names, hybrids, precedence_matrix=gac.cluster_precedence_matrix, edges=cluster_edges) 
    
    # Rematching with SecStrAnnotator
    if secstrannotator_rematching:
        for i in range(3):
            lib_sh.cp(directory / 'consensus.sses.json', directory / f'consensus_{i}.sses.json')
            labels = lib_acyclic_clustering_simple.rematch_with_SecStrAnnotator(domains, directory, sses, offsets, f'--fallback {fallback}' if fallback is not None else '')
            n_clusters, labels, cluster_precedence_matrix = lib_acyclic_clustering_simple.relabel_without_gaps(labels, gac.cluster_precedence_matrix)

            with open(directory/'labels_SSAnnot.cached.tsv', 'w') as w:
                w.write('\n'.join( str(l) for l in labels ))
            
            lib.log('Found', n_clusters, 'clusters')
            table = table_by_pdb_and_label(offsets, n_clusters, labels)
            hybrids = make_hybrid_sses_from_table(table, sses)
            cluster_edges = lib_acyclic_clustering_simple.cluster_edges(edges, labels)
            write_clustered_sses(directory, domain_names, hybrids, precedence_matrix=cluster_precedence_matrix, edges=cluster_edges)
            sizes, means, variances, covariances, minor_axes = sse_coords_stats(hybrids)
            print('Sorted DAG:', lib_graphs.sort_dag(range(n_clusters), lambda i,j: cluster_precedence_matrix[i,j]))
            print('Sizes:', sizes)
            
            # Calculation of self-classification probabilities
            self_probs, class_self_probs, total_self_prob = lib_acyclic_clustering_simple.self_classification_probabilities(coordinates, type_vector, labels, sse_weights=segment_lengths)
            print('\nClass self probabilities:')
            print(*( f'{i}: {p:.3f}' for i, p in enumerate(class_self_probs) ), sep='\n')
            print('\nTotal self probability:')
            print(f'{total_self_prob:.3f}')

            # agreement_strict = lib_acyclic_clustering_simple.labelling_agreement(no_rematch_labels, labels)
            # agreement_best = lib_acyclic_clustering_simple.labelling_agreement(no_rematch_labels, labels, allow_matching=True)
            # agreement_best_wo_unclass = lib_acyclic_clustering_simple.labelling_agreement(no_rematch_labels, labels, allow_matching=True, include_both_unclassified=False)
            # print('Agreement (no rematch vs ssan-rematch):', agreement_strict, agreement_best, agreement_best_wo_unclass)

    return


#  MAIN  #####################################################################################

@cli_command()
def main(directory: Path, secstrannotator_rematching: bool = False, fallback: Optional[float] = None) -> None:
    '''Perform clustering of SSEs from a set of domains, preserving SSE order and type.
    @param  `directory`       Directory with sample.json and structure files.
    @param  `secstrannotator_rematching`  Run 1 iteration of final rematching with SecStrAnnotator.
    @param  `fallback`        Parameter "fallback" for SecStrAnnotator.
    '''
    domains = lib_domains.load_domain_list(directory/'sample.json')
    lib.log(len(domains), 'domains')

    run_guided_clustering(domains, directory, secstrannotator_rematching=secstrannotator_rematching, fallback=fallback)

    # # REMATCH WITH SECSTRANNOTATOR
    # lib_sses.annotate_all_with_SecStrAnnotator(samples, directory)


if __name__ == '__main__':
    run_cli_command(main)
