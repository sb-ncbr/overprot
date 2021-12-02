'''
Performs clustering of SSEs from a set of domains, preserving SSE order and type.
'''

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import argparse
import itertools
from typing import Optional, Literal, Dict, Any, List

from .libs import lib
from .libs import lib_sh
from .libs import lib_graphs
from .libs import lib_domains
from .libs.lib_domains import Domain
from .libs import lib_sses
from .libs import lib_clustering
from .libs import lib_acyclic_clustering_simple
from .libs.lib_logging import ProgressBar

#  CONSTANTS  ################################################################################

QUASI_INFINITY = 1e9

NORM_DISTANCE_BY_SIZE = True
NORM_DISTANCE_MIN_SIZE = 3  # normalized distance is computed as dist / (size + NORM_DISTANCE_MIN_SIZE) (to avoid division by zero)
SIZE_PENALTY_FACTOR = 1
DUPLICITY_THRESHOLD = 0.6
CHAIN_ID_IN_CONSENSUS = 'A'
START_IN_CONSENSUS = 0
END_IN_CONSENSUS = 0

ENABLE_CACHED_LABELS = False  # True


#  FUNCTIONS  ################################################################################

def combine_distance_matrices(samples, directory: Path, length_thresholds=None, protein_similarity_weight=0):
    offsets = []
    sse_labels: list[str] = []
    sse_indices_in_domain = []
    n = len(samples)
    with ProgressBar(n + n*(n-1)//2, title='Combining distance matrices') as bar:
        for pdb, domain, chain, rang in samples:
            offsets.append(len(sse_labels))
            with open(directory / f'{domain}.sses.json') as f:
                these_sses = json.load(f)[domain]['secondary_structure_elements']
            if length_thresholds != None:
                selected_indices = lib.find_indices_where(these_sses, lambda s: lib_sses.long_enough(s, length_thresholds))
                these_sses  = [these_sses[i] for i in selected_indices]
            else:
                selected_indices = range(len(these_sses))
            sse_labels.extend((domain + '_' + s['label'] for s in these_sses))
            sse_indices_in_domain.extend(selected_indices)
            bar.step()
        N_sses = len(sse_labels)
        offsets.append(N_sses)
        #lib.log('Offsets:', offsets)

        supermatrix = np.zeros((N_sses, N_sses))

        # Fill same-domain fields
        pass # They will never be used

        # Fill different-domain fields
        DOMAIN = 1 # index of domain name in tuples in samples
        for i, j in itertools.combinations_with_replacement(range(len(samples)), 2):
            filename = directory / f'matrix-{samples[i][DOMAIN]}-{samples[j][DOMAIN]}.tsv'
            full_ij_matrix, *_ = lib.read_matrix(filename)
            selected_rows = sse_indices_in_domain[offsets[i]:offsets[i+1]]
            selected_columns = sse_indices_in_domain[offsets[j]:offsets[j+1]]
            ij_matrix = lib.submatrix_int_indexing(full_ij_matrix, selected_rows, selected_columns)
            col_mins = np.min(np.abs(ij_matrix), 0)
            row_mins = np.min(np.abs(ij_matrix), 1)
            # print(col_mins)
            # print(row_mins)
            min_distances = np.concatenate((col_mins, row_mins))
            protein_similarity = np.mean(min_distances)
            # print('Protein similarity ', protein_similarity)
            ij_matrix = ij_matrix + protein_similarity * protein_similarity_weight
            supermatrix[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] = ij_matrix
            supermatrix[offsets[j]:offsets[j+1], offsets[i]:offsets[i+1]] = ij_matrix.transpose()
            lib_sh.rm(filename, ignore_errors=True)
            bar.step()

    # Print the resulting matrix to a file
    lib.print_matrix(supermatrix, directory/'distance_matrix.tsv', sse_labels, sse_labels)

def read_sses(samples, directory: Path, length_thresholds=None):
    # if length_thresholds is not None and length_thresholds != {'H':0,'E':0}:
    # 	raise NotImplementedError()
    p_offsets = []  # counting a beta-ladder side as one SSE (for preference matrix)
    d_offsets = []  # counting a beta ladder as one SSE (for distance matrix)
    p_sses: list[dict] = []
    p_duplicates = []
    d_to_p: list[int|tuple[int,int]] = []  # pair of sides for a ladder / helix for a helix
    ladder_orientations = []  # 1 for parallel, -1 for antiparallel, 0 for helix
    d_helices = []
    d_ladders = []
    for pdb, domain, chain, rang in samples:
        p_offsets.append(len(p_sses))
        d_offsets.append(len(d_to_p))
        with open(directory/f'{domain}.sses.json') as f:
            annot = json.load(f)[domain]
        sses = annot['secondary_structure_elements']
        connectivity = annot['beta_connectivity']
        for sse in sses:
            sse['domain'] = domain
        # Filter SSEs by length:
        sses = [sse for sse in sses if lib_sses.long_enough(sse, length_thresholds)]
        labels = [sse['label'] for sse in sses]
        connectivity = lib.unique(conn for conn in connectivity if conn[0] in labels and conn[1] in labels)
        ladder_counter = lib.Counter[str]() # number of ladders for each strand
        for side1, side2, orientation in connectivity:
            ladder_counter.add(side1)
            ladder_counter.add(side2)
        # n_ladders = ladder_counter.all_counts()
        strand_name_lookup = {}
        for sse in sses:
            typ = lib_sses.two_class_type(sse)
            if typ == 'H':
                d_helices.append(len(d_to_p))
                d_to_p.append(len(p_sses))
                p_sses.append(sse)
                ladder_orientations.append(0)
            elif typ == 'E':
                label = sse['label']
                strand_name_lookup[label] = len(p_sses)
                n_duplicates = ladder_counter.count(label) # n_ladders[label]
                if n_duplicates > 1:
                    p_duplicates.append([len(p_sses) + i for i in range(n_duplicates)])
                for i in range(n_duplicates):
                    p_sses.append(sse)
        for side1, side2, orientation in connectivity:
            index1 = strand_name_lookup[side1]
            index2 = strand_name_lookup[side2]
            if index2 < index1:
                index1, index2 = index2, index1
            d_ladders.append(len(d_to_p))
            d_to_p.append((index1, index2))
            strand_name_lookup[side1] += 1  # next time use the next side
            strand_name_lookup[side2] += 1
            ladder_orientations.append(orientation)
    p_offsets.append(len(p_sses))
    d_offsets.append(len(d_to_p))
    # lib.log('p_offsets', p_offsets)
    # lib.log('d_offsets', d_offsets)
    # lib.log('p_sses', len(p_sses))
    # lib.log('d_to_p\n', d_to_p)
    # lib.log('orientation\n', ladder_orientations)

    n_D = len(d_to_p)
    n_P = len(p_sses)

    lib.log('Making precedence matrix...')
    precedence = np.zeros((n_P, n_P), dtype=np.bool_)
    for p in range(len(samples)):
        for i, j in itertools.combinations(range(p_offsets[p], p_offsets[p+1]), 2):
            if p_sses[i] != p_sses[j]:  # filter out cases when a strwith_cealignides
                precedence[i, j] = True

    lib.log('Extracting coordinates...')
    coordinates = np.zeros((n_D, 12), dtype=np.float64)
    for i in d_helices:
        ip = d_to_p[i]
        assert isinstance(ip, int)
        helix = p_sses[ip]
        coordinates[i,:] = helix['start_vector'] + helix['end_vector'] + [0]*6
    for i in d_ladders:
        ips = d_to_p[i]
        assert isinstance(ips, tuple)
        p1, p2 = ips
        side1, side2 = p_sses[p1], p_sses[p2]
        coordinates[i,:] = side1['start_vector'] + side1['end_vector'] + side2['start_vector'] + side2['end_vector']

    n_helices = len(d_helices)
    n_ladders = len(d_ladders)
    n_combinations = n_helices * (n_helices - 1) // 2 + n_ladders * (n_ladders - 1) // 2
    with ProgressBar(n_combinations, title='Making distance matrix...') as bar:
        distance = np.full((n_D, n_D), QUASI_INFINITY, dtype=np.float64)
        for i, j in itertools.combinations(d_helices, 2):
            dist = sse_distance(coordinates[i,0:6], coordinates[j,0:6])
            distance[i,j] = dist
            distance[j,i] = dist
            # lib.log('H', dist)
            bar.step()
        for i, j in itertools.combinations(d_ladders, 2):
            dist1 = sse_distance(coordinates[i,0:6], coordinates[j,0:6])  # distance of first sides
            dist2 = sse_distance(coordinates[i,6:12], coordinates[j,6:12])  # distance of second sides
            dist = 0.5 * (dist1 + dist2)
            distance[i,j] = dist
            distance[j,i] = dist
            # lib.log('E', dist)
            bar.step()

    lib.log('Printing matrices...')
    lib.print_matrix(distance, directory/'distance.tsv')
    lib.print_matrix(precedence, directory/'precedence.tsv')

    return p_offsets, d_offsets, p_sses, d_to_p, ladder_orientations, coordinates, distance, precedence, p_duplicates

def mean_of_min_protein_distance(sse_distance_matrix, offsets):
    pass

def sse_distance(coords1, coords2):
        diff = coords1 - coords2
        x1, y1, z1, x2, y2, z2 = diff
        dist = np.sqrt(x1*x1 + y1*y1 + z1*z1) + np.sqrt(x2*x2 + y2*y2 + z2*z2)
        dx, dy, dz = coords1[0:3] - coords1[3:6]
        size_i = np.sqrt(dx*dx + dy*dy + dz*dz)  # size of SSE i
        dx, dy, dz = coords2[0:3] - coords2[3:6]
        size_j = np.sqrt(dx*dx + dy*dy + dz*dz)  # size of SSE j
        if SIZE_PENALTY_FACTOR != 0:
            size_difference = abs(size_i - size_j)
            dist += SIZE_PENALTY_FACTOR * size_difference
        if NORM_DISTANCE_BY_SIZE:
            size = min(size_i, size_j)
            factor = 1.0 / (size + NORM_DISTANCE_MIN_SIZE)  #  + NORM_DISTANCE_MIN_SIZE to avoid division by zero
            dist = factor * dist
        return dist

def make_precedence_matrix(names):
    n = len(names)
    precedence = np.zeros((n, n), dtype=bool)
    pdbs, offsets = get_offsets(names, key=lambda name: name.split('_')[0])
    for p in range(len(pdbs)):
        for i, j in itertools.combinations(range(offsets[p], offsets[p+1]), 2):
            precedence[i, j] = True
    return precedence

def write_clustered_sses(directory: Path, domain_names: List[str], sse_table, precedence_matrix=None, edges=None):
    sizes, means, variances, covariances = sse_coords_stats(sse_table)
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
        consensus_sses.append(sse)
    consensus: Dict[str, Dict[str, Any]] = {'consensus': {'secondary_structure_elements': consensus_sses}}
    if edges is not None:
        beta_connectivity = [ (new_labels[s1], new_labels[s2], typ) for s1, s2, typ in edges ]
        sheets = lib_graphs.connected_components(None, edges)
        for i_sheet, sheet in enumerate(sheets):
            for i_strand in sheet:
                consensus_sses[i_strand]['sheet_id'] = i_sheet + 1
        consensus['consensus']['beta_connectivity'] = beta_connectivity
    else:
        for sse in consensus_sses:
            if sse is not None and sse['label'].startswith('E'):
                sse['sheet_id'] = 1
    lib.dump_json(consensus, directory/'consensus.sses.json')

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
                # print('Joined SSE:', sse)
                raise Exception(f'Yomummafat {i}, {j}.')
    return hybrids

def sse_coords_stats(hybrids):
    n_domains = len(hybrids)
    n_sses = len(hybrids[0])
    sizes, means, variances, covariances = [], [], [], []
    for i in range(n_sses):
        sses = [ hybrids[j][i] for j in range(n_domains) if hybrids[j][i] is not None ]
        coords = np.zeros((6, len(sses)), dtype=float)
        for j, sse in enumerate(sses):
            coords[0:3, j] = sse['start_vector']
            coords[3:6, j] = sse['end_vector']
        # mean = np.mean(coords, axis=1)
        mean = lib.safe_mean(coords, axis=1)
        cov = np.cov(coords) if len(sses) > 1 else np.zeros((6,6), dtype=float)
        var = np.trace(cov)
        # print(i)
        # print('Size', len(sses))
        # print('Mean', mean)
        # print('Var', var)
        # print('Cov', cov)
        # print()
        sizes.append(len(sses))
        means.append(mean)
        variances.append(var)
        covariances.append(cov)
    return sizes, means, variances, covariances

def calculate_duplicity(n_duplicates, n_1, n_2):
    duplicity = n_duplicates / min(n_1, n_2)
    # duplicity = n_duplicates / max(n_1, n_2)
    return duplicity


class AcyclicClusteringWithSides:
    def __init__(self, aggregate_function=lib_clustering.average_linkage_aggregate_function, max_joining_distance=np.inf):
        # aggregate_function - how to compute a value (of distance) for the cluster joined from two clusters value1,size1,value2,size2 => result_value
        self.aggregate_function = aggregate_function
        self.max_joining_distance = max_joining_distance

    def fit(self, distance_matrix, precedence_matrix, d_to_p, type_vector=None, domains=None, p_offsets=None, member_count_threshold=0):
        # only samples with the same type can be linked
        n_D = distance_matrix.shape[0] # number of samples = number of leaves
        m = 2*n_D - 1 # max. possible number of all nodes
        if distance_matrix.shape[1] != n_D:
            raise Exception('distance_matrix must be a square matrix')
        n_P = precedence_matrix.shape[0] # number of samples = number of leaves
        if precedence_matrix.shape[1] != n_P:
            raise Exception('precedence_matrix must be a square matrix')
        curr_n_nodes = n_D
        leader = list(range(n_D))
        members = [[i] for i in range(n_D)]
        children = np.full((m, 2), -1) # matrix of children pairs, -1 = no child
        self.distances = np.full(m, 0.0) # distances between the children of each internal node
        self.children = children
        active_nodes = set(range(n_D))
        D = distance_matrix # no copying (to spare memory)
        P = precedence_matrix # no copying (to spare memory)
        T = type_vector if type_vector is not None else np.zeros(n_D, dtype=int) # type vector

        def can_join(ij):
            i, j = ij
            if i in active_nodes and j in active_nodes:
                li = leader[i]
                lj = leader[j]
                if T[li] == T[lj]:
                    if T[li] == 0:  # helices
                        pi = d_to_p[li]
                        pj = d_to_p[lj]
                        return not P[pi, pj] and not P[pj, pi]
                    else:  # ladders (both parallel or both antiparallel)
                        pi1, pi2 = d_to_p[li]  # two sides of the ladder li
                        pj1, pj2 = d_to_p[lj]
                        return not P[pi1, pj1] and not P[pj1, pi1] and not P[pi2, pj2] and not P[pj2, pi2]
            return False

        # distance_queue = lib.PriorityQueue(( ((i, j), D[i, j]) for (i, j) in itertools.combinations(active_nodes, 2) if can_join((i, j)) ))
        distance_queue = lib.PriorityQueue(( (D[i, j], (i, j)) for (i, j) in itertools.combinations(active_nodes, 2) if can_join((i, j)) ))

        #print('D:', D.shape, 'P:', P.shape)
        with ProgressBar(n_D-1, title='Clustering ' + str(n_D) + ' samples') as progress_bar:
            while len(active_nodes) >= 2:
                best_pair = distance_queue.pop_min_which(can_join)
                if best_pair == None:
                    break  # no joinable pairs, algorithm has converged
                distance, (p, q) = best_pair
                if distance > self.max_joining_distance:
                    break  # no joinable pairs with lower distance than the limit, algorithm has converged
                p_leader, q_leader = leader[p], leader[q]
                new_leader = min(p_leader,  q_leader)
                new = curr_n_nodes
                curr_n_nodes += 1
                leader.append(p_leader)
                members.append(members[p] + members[q])
                children[new,:] = [p, q]
                self.distances[new] = distance
                active_nodes.remove(p)
                active_nodes.remove(q)
                active_nodes.add(new)

                # Update precedence matrix (apply transitivity)
                #TODO remember p_active_leaders in a set variable
                d_active_leaders = [leader[i] for i in active_nodes]
                p_active_leaders = []
                for i in d_active_leaders:
                    if T[i] == 0:  # helix
                        p_active_leaders.append(d_to_p[i])
                    else:  # ladder
                        p_active_leaders.extend(d_to_p[i])
                if T[new_leader] == 0:  # helix
                    p_p = d_to_p[p_leader]
                    p_q = d_to_p[q_leader]
                    self.update_precedence(P, p_active_leaders, p_p, p_q)
                else:  # ladder
                    p_p1, p_p2 = d_to_p[p_leader]
                    p_q1, p_q2 = d_to_p[q_leader]
                    self.update_precedence(P, p_active_leaders, p_p1, p_q1)
                    self.update_precedence(P, p_active_leaders, p_p2, p_q2)
                # Update distance matrix (calculate distances from the new cluster)
                for i in active_nodes:
                    if i != new:
                        i_leader = leader[i]
                        D[new_leader, i_leader] = self.aggregate_function(D[p_leader, i_leader], len(members[p]), D[q_leader, i_leader], len(members[q]))
                        D[i_leader, new_leader] = D[new_leader, i_leader]
                        if can_join((i,new)):
                            distance_queue.add(D[new_leader, i_leader], (i, new))
                progress_bar.step()

        # Filter by cluster size (number of members)
        # active_nodes = [node for node in active_nodes if len(members[node]) >= member_count_threshold]
        # TODO uncomment and debug this

        halfnodes_to_sort = []  # tuples (node, side 0/1)
        p_leaders_to_sort = {}
        edges = []
        for i in active_nodes:
            l = leader[i]
            if T[l] == 0:  # helix
                halfnodes_to_sort.append((i, 0))
                p_leaders_to_sort[(i, 0)] = d_to_p[l]
            else:  # ladder
                halfnodes_to_sort.extend([(i, 0), (i, 1)])
                p_leaders_to_sort[(i, 0)] = d_to_p[l][0]
                p_leaders_to_sort[(i, 1)] = d_to_p[l][1]
        sorted_halfnodes = lib.sort_dag(halfnodes_to_sort, lambda i, j: P[p_leaders_to_sort[i], p_leaders_to_sort[j]])
        sorted_nodes = [node for (node, side) in sorted_halfnodes if side==0]
        node_to_zeroth_halfnode_index = { node: idx for idx, (node, side) in enumerate(sorted_halfnodes) if side == 0 }
        self.edges = []
        for idx, (node, side) in enumerate(sorted_halfnodes):
            if side == 1:
                other_idx = node_to_zeroth_halfnode_index[node]
                typ = T[leader[node]]
                self.edges.append((other_idx, idx, typ))
        self.edges.sort(key=lambda e: e[0])
        self.final_d_members = [members[node] for node in sorted_nodes]
        self.n_clusters = len(sorted_nodes)
        self.d_labels = np.zeros(n_D, dtype=int)
        for label in range(self.n_clusters):
            self.d_labels[self.final_d_members[label]] = label
        sorted_active_leaders = [leader[node] for node in sorted_nodes]
        self.cluster_distance_matrix = lib.submatrix_int_indexing(D, sorted_active_leaders, sorted_active_leaders)
        self.final_p_members = []
        cluster_p_leaders = []
        for node, side in sorted_halfnodes:
            typ = T[leader[node]]
            if typ == 0:  # helix
                p_members = [d_to_p[mem] for mem in members[node]]
                self.final_p_members.append(p_members)
                cluster_p_leaders.append(d_to_p[leader[node]])
            else:  # side
                p_members = [d_to_p[mem][side] for mem in members[node]]
                self.final_p_members.append(p_members)
                cluster_p_leaders.append(d_to_p[leader[node]][side])
        self.p_labels = np.zeros(n_P, dtype=int)
        for label in range(len(self.final_p_members)):
            self.p_labels[self.final_p_members[label]] = label
        self.n_p_clusters = len(sorted_halfnodes)
        self.cluster_precedence_matrix = np.zeros((self.n_p_clusters, self.n_p_clusters), dtype=bool)
        for i, p_leader_i in enumerate(cluster_p_leaders):
            for j, p_leader_j in enumerate(cluster_p_leaders):
                self.cluster_precedence_matrix[i, j] = P[p_leader_i, p_leader_j]


        lib.log('Sorted P leaders', sorted_halfnodes)
        lengths = np.zeros((len(sorted_halfnodes), len(domains)), dtype=int)
        names = []
        inv_p_offsets = lib.invert_offsets(p_offsets)
    
    def group_p_duplicates(self, p_duplicates, duplicity_threshold):
        had_achtung = False
        duplicity_counter = lib.Counter()
        for dups in p_duplicates:
            for p1, p2 in itertools.combinations(dups, 2):
                label1, label2 = sorted((self.p_labels[p1], self.p_labels[p2]))
                duplicity_counter.add((label1, label2))
        n_p = self.n_p_clusters
        p2leader = list(range(n_p))
        grouped_precedence = self.cluster_precedence_matrix.copy()
        for (lab1, lab2), n_duplicates in sorted(duplicity_counter.all_counts().items(), key=lambda t: t[1], reverse=True):
            n_members1 = len(self.final_p_members[lab1])
            n_members2 = len(self.final_p_members[lab2])
            duplicity = calculate_duplicity(n_duplicates, n_members1, n_members2)
            if duplicity >= duplicity_threshold:
                leader1 = p2leader[lab1]
                leader2 = p2leader[lab2]
                lib.log(f'Joining {leader1} and {leader2} ?', end=' ')
                # grouped_precedence[leader,:] = np.logical_or(grouped_precedence[leader,:], grouped_precedence[member,:])
                after1before2 = np.logical_and(grouped_precedence[leader1,:], grouped_precedence[:,leader2])
                after2before1 = np.logical_and(grouped_precedence[leader2,:], grouped_precedence[:,leader1])
                if any(after1before2) or any(after2before1):
                    lib.log('Not today!')
                    pass  # this joining would break the ordering -> skip joining
                else:
                    lib.log('Yes!')
                    grouped_precedence[leader1,:] = np.logical_or(grouped_precedence[leader1,:], grouped_precedence[leader2,:])
                    grouped_precedence[:,leader1] = np.logical_or(grouped_precedence[:,leader1], grouped_precedence[:,leader2])
                    grouped_precedence[leader1,leader1] = False
                    p2leader[lab2] = leader1
        leaders = set(p2leader)
        leaders = lib.sort_dag(leaders, lambda l1, l2: grouped_precedence[l1, l2])
        leader2group = { leader: i for i, leader in enumerate(leaders) }
        p2group = [ leader2group[l] for l in p2leader ]

        self.n_g_clusters = len(leaders)
        self.g_precedence = lib.submatrix_int_indexing(grouped_precedence, leaders, leaders)
        # Map original p_labels to grouped labels (g_labels)
        self.g_labels = [ p2group[p] for p in self.p_labels ]
        self.g_edges = [ (p2group[p1], p2group[p2], typ) for p1, p2, typ in self.edges ]
        self.g_edges.sort(key=lambda e: e[0])


    def group_p_duplicates_old(self, p_duplicates, duplicity_threshold):
        had_achtung = False
        duplicity_counter = lib.Counter()
        for dups in p_duplicates:
            for p1, p2 in itertools.combinations(dups, 2):
                label1, label2 = sorted((self.p_labels[p1], self.p_labels[p2]))
                duplicity_counter.add((label1, label2))
        n_sses = len(self.final_p_members)
        labels = set(self.p_labels[sse] for dup in p_duplicates for sse in dup)
        groups = [[label] for label in labels]
        for (lab1, lab2), n_duplicates in sorted(duplicity_counter.all_counts().items(), key=lambda t: t[1], reverse=True):
            n_members1 = len(self.final_p_members[lab1])
            n_members2 = len(self.final_p_members[lab2])
            duplicity = calculate_duplicity(n_duplicates, n_members1, n_members2)
            lib.log(lab1, lab2, 'duplicates', n_duplicates, 'duplicity', format(duplicity, '1.2f'))
            if duplicity >= duplicity_threshold:
                group1 = next(group for group in groups if lab1 in group)
                group2 = next(group for group in groups if lab2 in group)
                lib.log('joining', group1, 'and', group2)
                if group1 != group2:
                    groups.remove(group2)
                    group1.extend(group2)
                    lib.log('  =>', group1)
        groups.sort()
        lib.log('groups for joining:', *groups)
        # Calculate new precedence matrix
        grouped_precedence = self.cluster_precedence_matrix.copy()
        for i, group in enumerate(groups):
            if len(group) > 1:
                leader = min(group)
                # Propagation of precedence within group
                for member in group:
                    if member != leader:
                        grouped_precedence[leader,:] = np.logical_or(grouped_precedence[leader,:], grouped_precedence[member,:])
                        grouped_precedence[:,leader] = np.logical_or(grouped_precedence[:,leader], grouped_precedence[:,member])
                # Annihilation of opposite precedence within group (what is after cannot be before)
                is_before_and_after = np.logical_and(grouped_precedence[:,leader], grouped_precedence[leader,:])
                is_before_and_after[group] = False
                before_and_after = np.flatnonzero(is_before_and_after)
                # print(i, '(', len(group), ') Before and after:', is_before_and_after.shape, before_and_after)
                for p in before_and_after:
                    print('ACHTUNG:', p, 'is before and after', *group)
                    had_achtung = True
                    grouped_precedence[leader,p] = False
                    grouped_precedence[p, leader] = False
                grouped_precedence[leader, leader] = False
        groups.extend( [lab] for lab in range(n_sses) if lab not in labels )  # Add p_clusters which were not subject of grouping
        groups = lib.sort_dag(groups, lambda g1, g2: grouped_precedence[min(g1), min(g2)])
        self.n_g_clusters = len(groups)
        group_leaders = [min(group) for group in groups]
        self.g_precedence = lib.submatrix_int_indexing(grouped_precedence, group_leaders, group_leaders)
        # Map original p_labels to grouped labels (g_labels)
        p_to_g = [0] * n_sses
        for i, group in enumerate(groups):
            for member in group:
                p_to_g[member] = i
        self.g_labels = [ p_to_g[p] for p in self.p_labels ]
        self.g_edges = [ (p_to_g[p1], p_to_g[p2], typ) for p1, p2, typ in self.edges ]
        self.g_edges.sort(key=lambda e: e[0])
        lib.log('HAD ACHTUNG' if had_achtung else 'KEINE ACHTUNG')
        print('BH')

    @staticmethod
    def update_precedence(P, p_active_leaders, p, q):  # joins clusters p and q in precedence matrix P, with active leaders p_active_leaders (+ formerly p and q)
        to_p = [l for l in p_active_leaders if P[l, p] and not P[l, q]] + [p]
        to_q = [l for l in p_active_leaders if P[l, q] and not P[l, p]] + [q]
        from_p = [l for l in p_active_leaders if P[p, l] and not P[q, l]] + [p]
        from_q = [l for l in p_active_leaders if P[q, l] and not P[p, l]] + [q]
        for i in to_p:
            for j in from_q:
                P[i, j] = True
        for i in to_q:
            for j in from_p:
                P[i, j] = True


def run_simple_clustering(domains: List[Domain], directory: Path, min_occurrence: float = 0.0, secstrannotator_rematching: bool = False, fallback: Optional[float] = None) -> None:
    domain_names = [dom.name for dom in domains]
    offsets, sses, coordinates, type_vector, edges = lib_acyclic_clustering_simple.read_sses_simple(domains, directory)
    n_structures = len(offsets) - 1
    segment_lengths = lib_acyclic_clustering_simple.segment_lengths(coordinates)
    if 0 in segment_lengths:
        raise Exception('Some of the SSE segment lengths are zero. Use newer version of secondary structure assignment with non-zero lengths.')

    labels = None
    cache_file = directory/'labels_orig.cached.tsv'

    # Try to load labels from a cache-file
    if ENABLE_CACHED_LABELS and cache_file.is_file():
        with open(cache_file) as r:
            labels = np.array([ int(l) for l in r.read().split() ])
        n_clusters = max(labels) + 1
        cluster_precedence_matrix, _, _ = lib.read_matrix(directory/'cluster_precedence_matrix.tsv')
        if len(labels) == len(sses):
            print(f'\nLabels loaded from cache file "{cache_file}", no clustering performed!\n')
        else:  # Cancel the loaded labels, they are outdated.
            print(f'\nCached labels from "{cache_file}" are outdated.\n')
            labels = None
    elif not ENABLE_CACHED_LABELS:
        print(f'\nLoading cached labels is disabled.\n')
    else:
        print(f'\nCache file {cache_file} does not exist.\n')
    
    # Run clustering if label-caching is disabled or cached labels are invalid
    if labels is None:
        precedence = lib_acyclic_clustering_simple.make_precedence_matrix(offsets)
        distance = lib_acyclic_clustering_simple.sse_distance_matrix(coordinates)

        # distance += lib_acyclic_clustering_simple.segment_length_difference_matrix(coordinates)  # length-difference penalty
        distance = lib_acyclic_clustering_simple.include_min_ladder_in_distance_matrix(distance, edges)
        lib.print_matrix(distance, directory/'distance.tsv')

        distance_iter = lib_acyclic_clustering_simple.distance_matrix_with_iterative_superimposition_many(coordinates, type_vector, offsets, edges=edges)
        lib.print_matrix(distance_iter, directory/'distance_iter.tsv')

        distance = distance_iter

        DYNPROG_SCORE_MAX = 30
        dynprog_score = lib_acyclic_clustering_simple.linearly_decreasing_score(distance, type_vector, intercept=DYNPROG_SCORE_MAX)
        dynprog_score *= lib.each_to_each(np.minimum, segment_lengths)
        fuckup = lib_acyclic_clustering_simple.dynprog_fuckup_indices_each_to_each(dynprog_score, offsets)

        # total_dynprog_scores = lib_acyclic_clustering_simple.dynprog_total_scores_each_to_each(dynprog_score, offsets)
        # dynprog_norm_coeff = 1 / np.sqrt(total_dynprog_scores.diagonal())
        # total_dynprog_scores_norm = total_dynprog_scores * lib.each_to_each(np.multiply, dynprog_norm_coeff)
        # protein_distance = (total_dynprog_scores_norm.max() - total_dynprog_scores_norm) / (total_dynprog_scores_norm.max() - np.median(total_dynprog_scores_norm))
        # sse_distance_norm_coeff = 1 / np.median(distance_ladders)

        distance_final = distance + 1 * fuckup

        #region Plots
        # from matplotlib import pyplot as plt
        # plt.plot(sorted(distance_final.flatten()), '.')
        # plt.show()
        # plt.plot(sorted(distance_ladders.flatten()), '.')
        # plt.show()
        # plt.plot(sorted(total_dynprog_scores.flatten()), '.')
        # plt.show()
        # plt.plot(sorted(total_dynprog_scores_norm.flatten()), '.')
        # plt.show()
        # plt.plot(sorted(protein_distance.flatten()), '.')
        # plt.show()
        # plt.plot(sorted(sse_distance_norm_coeff * distance_ladders.flatten()), '.')
        # plt.show()
        # plt.plot(sorted(sse_distance_norm_coeff * distance_final.flatten()), '.')
        # plt.show()
        # plt.plot(sorted(1/total_dynprog_scores.flatten(), reverse=True), '.')
        # plt.show()
        # plt.plot(sorted(total_dynprog_scores.diagonal()), '.')
        # plt.show()
        # exit(0)
        #endregion

        # distance_final = lib_acyclic_clustering_simple.include_protein_distance(sse_distance_norm_coeff * distance_final, protein_distance, offsets, overwrite=True) 

        distance_final /= lib.each_to_each(np.minimum, segment_lengths)

        acs = lib_acyclic_clustering_simple.AcyclicClusteringSimple(aggregate_function=lib_clustering.average_linkage_aggregate_function,  max_joining_distance=np.inf)
        acs.fit(distance_final, precedence, type_vector=type_vector, domain_names=domain_names, p_offsets=offsets, member_count_threshold=0, output_dir=directory)

        lib.log('Found', acs.n_clusters, 'clusters')
        table = table_by_pdb_and_label(offsets, acs.n_clusters, acs.labels)
        hybrids = make_hybrid_sses_from_table(table, sses)
        cluster_edges = lib_acyclic_clustering_simple.cluster_edges(edges, acs.labels)
        # print('Table:', table)
        write_clustered_sses(directory, domain_names, hybrids, precedence_matrix=acs.cluster_precedence_matrix, edges=cluster_edges) 
        sizes, means, variances, covariances = sse_coords_stats(hybrids)
        print('Sorted DAG:', lib_graphs.sort_dag(range(acs.n_clusters), lambda i,j: acs.cluster_precedence_matrix[i,j]))
        with open(cache_file, 'w') as w:
            w.write('\n'.join( str(l) for l in acs.labels ))
        
        labels = acs.labels
        n_clusters = acs.n_clusters
        cluster_precedence_matrix = acs.cluster_precedence_matrix

    no_rematch_labels = labels.copy()
    print('ANOVA F:', lib_acyclic_clustering_simple.anova_f(coordinates, labels))


    # Iterative rematching (1 run with non-adhesive iter. rematching, then 1 run with adhesive iter. rematching)
    # labels = lib_acyclic_clustering_simple.iterative_rematching(coordinates, type_vector, labels, edges, offsets, adhesion=0, max_iterations=np.inf)
    labels = lib_acyclic_clustering_simple.iterative_rematching(coordinates, type_vector, labels, edges, offsets, adhesion=1, max_iterations=np.inf)
    labels = lib_acyclic_clustering_simple.filter_labels_by_count(labels, int(min_occurrence * n_structures))
    n_clusters, labels, cluster_precedence_matrix = lib_acyclic_clustering_simple.relabel_without_gaps(labels, cluster_precedence_matrix)
    print('ANOVA F:', lib_acyclic_clustering_simple.anova_f(coordinates, labels))

    with open(directory/'labels_rematched.cached.tsv', 'w') as w:
        w.write('\n'.join( str(l) for l in labels ))

    lib.log('Found', n_clusters, 'clusters')
    table = table_by_pdb_and_label(offsets, n_clusters, labels)
    hybrids = make_hybrid_sses_from_table(table, sses)
    cluster_edges = lib_acyclic_clustering_simple.cluster_edges(edges, labels)
    # print('Table:', table)
    # print('Hybrids:', hybrids)
    write_clustered_sses(directory, domain_names, hybrids, precedence_matrix=cluster_precedence_matrix, edges=cluster_edges)
    sizes, means, variances, covariances = sse_coords_stats(hybrids)
    print('Sorted DAG:', lib_graphs.sort_dag(range(n_clusters), lambda i,j: cluster_precedence_matrix[i,j]))
    print('Sizes:', sizes)
    
    # Calculation of self-classification probabilities
    # self_probs, class_self_probs, total_self_prob = lib_acyclic_clustering_simple.self_classification_probabilities(coordinates, type_vector, labels)
    self_probs, class_self_probs, total_self_prob = lib_acyclic_clustering_simple.self_classification_probabilities(coordinates, type_vector, labels, sse_weights=segment_lengths)
    print('\nClass self probabilities:')
    print(*( f'{i}: {p:.3f}' for i, p in enumerate(class_self_probs) ), sep='\n')
    print('\nTotal self probability:')
    print(f'{total_self_prob:.3f}')

    agreement_strict = lib_acyclic_clustering_simple.labelling_agreement(no_rematch_labels, labels)
    agreement_best = lib_acyclic_clustering_simple.labelling_agreement(no_rematch_labels, labels, allow_matching=True)
    agreement_best_wo_unclass = lib_acyclic_clustering_simple.labelling_agreement(no_rematch_labels, labels, allow_matching=True, include_both_unclassified=False)
    print('Agreement (no rematch vs rematch):', agreement_strict, agreement_best, agreement_best_wo_unclass)


    # Rematching with SecStrAnnotator
    if secstrannotator_rematching:
        labels = lib_acyclic_clustering_simple.rematch_with_SecStrAnnotator(domains, directory, sses, offsets, f'--fallback {fallback}' if fallback is not None else '')
        n_clusters, labels, cluster_precedence_matrix = lib_acyclic_clustering_simple.relabel_without_gaps(labels, cluster_precedence_matrix)

        with open(directory/'labels_SSAnnot.cached.tsv', 'w') as w:
            w.write('\n'.join(str(l) for l in labels))
        
        lib.log('Found', n_clusters, 'clusters')
        table = table_by_pdb_and_label(offsets, n_clusters, labels)
        hybrids = make_hybrid_sses_from_table(table, sses)
        cluster_edges = lib_acyclic_clustering_simple.cluster_edges(edges, labels)
        # print('Table:', table)
        # print('Hybrids:', hybrids)
        write_clustered_sses(directory, domain_names, hybrids, precedence_matrix=cluster_precedence_matrix, edges=cluster_edges)
        sizes, means, variances, covariances = sse_coords_stats(hybrids)
        print('Sorted DAG:', lib_graphs.sort_dag(range(n_clusters), lambda i,j: cluster_precedence_matrix[i,j]))
        print('Sizes:', sizes)
        
        # Calculation of self-classification probabilities
        # self_probs, class_self_probs, total_self_prob = lib_acyclic_clustering_simple.self_classification_probabilities(coordinates, type_vector, labels)
        self_probs, class_self_probs, total_self_prob = lib_acyclic_clustering_simple.self_classification_probabilities(coordinates, type_vector, labels, sse_weights=segment_lengths)
        print('\nClass self probabilities:')
        print(*( f'{i}: {p:.3f}' for i, p in enumerate(class_self_probs) ), sep='\n')
        print('\nTotal self probability:')
        print(f'{total_self_prob:.3f}')

        agreement_strict = lib_acyclic_clustering_simple.labelling_agreement(no_rematch_labels, labels)
        agreement_best = lib_acyclic_clustering_simple.labelling_agreement(no_rematch_labels, labels, allow_matching=True)
        agreement_best_wo_unclass = lib_acyclic_clustering_simple.labelling_agreement(no_rematch_labels, labels, allow_matching=True, include_both_unclassified=False)
        print('Agreement (no rematch vs ssan-rematch):', agreement_strict, agreement_best, agreement_best_wo_unclass)

    lib_sses.map_manual_template_to_consensus(directory)


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
    For a==0, the function is equal to f(x) = max(1-x/k, 0)'''
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
    
    # for attr in ['n_clusters', 'final_members', 'labels', 'cluster_precedence_matrix']:
    #     lib.log_debug(attr, gac.__getattribute__(attr), sep='\n')

    table = table_by_pdb_and_label(offsets, gac.n_clusters, gac.labels)
    hybrids = make_hybrid_sses_from_table(table, sses)
    cluster_edges = lib_acyclic_clustering_simple.cluster_edges(edges, gac.labels)
    if any(u == v for u, v, *rest in cluster_edges):
        raise Exception(f'Resulting beta connectivity contains self-connections.')
        # cluster_edges = [(u, v, *rest) for u, v, *rest in cluster_edges if u != v]
    write_clustered_sses(directory, domain_names, hybrids, precedence_matrix=gac.cluster_precedence_matrix, edges=cluster_edges) 
    # sizes, means, variances, covariances = sse_coords_stats(hybrids)
    
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
            # print('Table:', table)
            # print('Hybrids:', hybrids)
            write_clustered_sses(directory, domain_names, hybrids, precedence_matrix=cluster_precedence_matrix, edges=cluster_edges)
            sizes, means, variances, covariances = sse_coords_stats(hybrids)
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


def run_clustering_with_sides(samples, directory: Path, protein_similarity_weight=0, length_thresholds=None):

    p_offsets, d_offsets, p_sses, d_to_p, type_vector, coordinates, distance_matrix, precedence_matrix, p_duplicates = read_sses(samples, directory, length_thresholds=length_thresholds)

    # Include global protein-protein similarity into the distance matrix
    if protein_similarity_weight != 0:
        protein_similarity_matrix, _, _ = lib.read_matrix(directory/'mapsci_score_matrix.tsv')
        mean_similarity = np.mean(protein_similarity_matrix.flatten())
        protein_similarity_matrix /= mean_similarity
        print('Max MASPCI score:', np.max(protein_similarity_matrix.flatten()))
        print('Mean MASPCI score:', np.mean(protein_similarity_matrix.flatten()))
        print('Median MASPCI score:', np.median(protein_similarity_matrix.flatten()))
        for i in range(len(d_offsets) - 1):
            for j in range(len(d_offsets) - 1):
                distance_matrix[d_offsets[i]:d_offsets[i+1], d_offsets[j]:d_offsets[j+1]] += protein_similarity_matrix[i, j] * protein_similarity_weight

    # Run clustering
    ac = AcyclicClusteringWithSides(aggregate_function=lib_clustering.average_linkage_aggregate_function, max_joining_distance=np.inf)
    ac.fit(distance_matrix, precedence_matrix, d_to_p, type_vector=type_vector, domains=None, p_offsets=p_offsets, member_count_threshold=0)

    # TODO debug filtering clusters by member_count_threshold + computation of length

    lib.log('Found', ac.n_clusters, 'clusters')
    ac.group_p_duplicates(p_duplicates, DUPLICITY_THRESHOLD)
    table = table_by_pdb_and_label(p_offsets, ac.n_g_clusters, ac.g_labels)
    hybrids = make_hybrid_sses_from_table(table, p_sses)
    print('Table:', table)
    print('Hybrids:', hybrids)
    write_clustered_sses(directory, [], hybrids, precedence_matrix=ac.g_precedence, edges=ac.g_edges)
    sizes, means, variances, covariances = sse_coords_stats(hybrids)
    print('G-edges:', *ac.g_edges, sep='\n')
    print('Sorted P-DAG:', lib_graphs.sort_dag(range(ac.n_p_clusters), lambda i,j: ac.cluster_precedence_matrix[i,j]))
    print('Sorted G-DAG:', lib_graphs.sort_dag(range(ac.n_g_clusters), lambda i,j: ac.g_precedence[i,j]))

    # TODO change SecStrAnnotator so it can read precedence matrix (ideally in some compressed form)
    # TODO allow filtering clusters by occurence threshold (be carefull with edges!)


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='Directory with sample.json and structure files', type=Path)
    parser.add_argument('--force_ssa', help='Run SecStrAnnotator to calculate secondary structure assignment files even if they already exist', action='store_true')
    parser.add_argument('--secstrannotator_rematching', help='Run 1 iteration of final rematching with SecStrAnnotator', action='store_true')
    parser.add_argument('--min_length_h', help='Minimal length of a helix to be considered', type=int, default=0)
    parser.add_argument('--min_length_e', help='Minimal length of a strand to be considered', type=int, default=0)
    parser.add_argument('--protsim', help='Protein similarity weight (w.r.t. to SSE similarity), default=0', type=float, default=0)
    parser.add_argument('--min_occurrence', help='Minimal occurrence of SSE cluster to be included in the result (0 to 1), default=0', type=float, default=0)
    parser.add_argument('--fallback', help='Parameter "fallback" for SecStrAnnotator', type=float, default=None)
    args = parser.parse_args()
    return vars(args)


def main(directory: Path,
         force_ssa: bool = False, secstrannotator_rematching: bool = False, min_length_h: int = 0, min_length_e: int = 0, 
         protsim: float = 0, min_occurrence: float = 0, fallback: Optional[float] = None) -> Optional[int]:
    '''Foo'''
    # TODO add docstring

    domains = lib_domains.load_domain_list(directory/'sample.json')
    lib.log(len(domains), 'domains')

    METHOD = 'guided'  # guided | simple | sides

    if METHOD == 'simple':
        # SIMPLE CLUSTERING
        run_simple_clustering(domains, directory, min_occurrence=min_occurrence, secstrannotator_rematching=secstrannotator_rematching, fallback=fallback)
    elif METHOD == 'guided':
        # GUIDED CLUSTERING
        run_guided_clustering(domains, directory, secstrannotator_rematching=secstrannotator_rematching, fallback=fallback)
    elif METHOD == 'sides':
        # CLUSTERING WITH SIDES
        run_clustering_with_sides(domains, directory, length_thresholds={'H': min_length_h, 'E': min_length_e}, protein_similarity_weight=protsim)
    else:
        raise AssertionError(f'Unknown clustering method: {METHOD}')

    # # REMATCH WITH SECSTRANNOTATOR
    # lib_sses.annotate_all_with_SecStrAnnotator(samples, directory)

    return None


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)
