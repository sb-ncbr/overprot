# Performs clustering of SSEs from a set of domains, preserving SSE order and type.
# Requires these files to be in the working directory:
#     cytos.exe
#     script_align.py

import json
import os
from os import path
import sys
import numpy as np
import argparse
import itertools
import heapq
import re
from collections import defaultdict

from ete3 import Tree

from libs import lib
from libs import lib_clustering

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('directory', help='Directory with sample.json and structure files', type=str)
parser.add_argument('--run_ssa', help='Run cytos.exe to calculate distance matrices and calculate precedence matrix', action='store_true')
parser.add_argument('--run_annotator', help='Run cytos.exe to calculate distance matrices and calculate precedence matrix', action='store_true')
parser.add_argument('--run_distance_from_alignment', help='Calculate distance matrices assuming multiple structural alignment', action='store_true')
parser.add_argument('--min_length_h', help='Minimal length of a helix to be considered', type=int, default=0)
parser.add_argument('--min_length_e', help='Minimal length of a strand to be considered', type=int, default=0)
#choice=, add_mutually_exclusive_group()
args = parser.parse_args()

if args.run_annotator and args.run_distance_from_alignment:
    raise Exception('Cannot combine --run_annotator and --run_distance_from_alignment')

QUASI_INFINITY = 1e6
COMMENT_SYMBOL = '#'
CYTOS_BINARY = 'SecStrAnnotator.exe'
RE_DTYPE = re.compile('^\s*'+re.escape(COMMENT_SYMBOL)+'\s*dtype\s*=\s*(\w+)\s*$')
RE_SHAPE = re.compile('^\s*'+re.escape(COMMENT_SYMBOL)+'\s*shape\s*=\s*(\w+)\s*,\s*(\w+)\s*$')

PROTEIN_SIMILARITY_WEIGHT = 0  # How much global protein-protein similarity is included in SSE-SSE distance
NORM_DISTANCE_BY_SIZE = True
NORM_DISTANCE_MIN_SIZE = 3  # normalized distance is computed as dist / (size + NORM_DISTANCE_MIN_SIZE) (to avoid division by zero)
SIZE_PENALTY_FACTOR = True

################################################################################

def combine_distance_matrices(samples, directory, length_thresholds=None, protein_similarity_weight=0):
    offsets = []
    sse_labels = []
    sse_indices_in_domain = []
    n = len(samples)
    progess_bar = ProgressBar(n + n*(n-1)/2, title='Combining distance matrices').start()
    for pdb, domain, chain, rang in samples:
        offsets.append(len(sse_labels))
        with open(path.join(directory, domain+'.sses.json')) as f:
            these_sses = json.load(f)[domain]['secondary_structure_elements']
        if length_thresholds != None:
            selected_indices = find_indices_where(these_sses, lambda s: long_enough(s, length_thresholds))
            these_sses  = [these_sses[i] for i in selected_indices]
        else:
            selected_indices = range(len(these_sses))
        sse_labels.extend((domain + '_' + s['label'] for s in these_sses))
        sse_indices_in_domain.extend(selected_indices)
        progess_bar.step()
    N_sses = len(sse_labels)
    offsets.append(N_sses)
    # lib.log('Offsets:', offsets)

    supermatrix = np.zeros((N_sses, N_sses))

    # Fill same-domain fields
    pass # They will never be used

    # Fill different-domain fields
    DOMAIN = 1 # index of domain name in tuples in samples
    for i, j in itertools.combinations_with_replacement(range(len(samples)), 2):
        filename = path.join(directory, 'matrix-'+samples[i][DOMAIN]+'-'+samples[j][DOMAIN]+'.tsv')
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
        run_command('rm', '-f', filename)
        progess_bar.step()
    progess_bar.finalize()
    
    # Print the resulting matrix to a file
    lib.print_matrix(supermatrix, path.join(directory, 'distance_matrix.tsv'), sse_labels, sse_labels)

def compute_whole_distance_matrix_without_alignment(samples, directory, length_thresholds=None, protein_similarity_weight=0, norm_distance_by_size=False, size_penalty_factor=0):
    offsets = []
    sse_labels = []
    sse_indices_in_domain = []
    sse_coordinates = []
    n = len(samples)
    progess_bar = ProgressBar(n, title='Computing distance matrices from multiple-alignment coordinates - preparation').start()
    for pdb, domain, chain, rang in samples:
        offsets.append(len(sse_labels))
        with open(path.join(directory, domain+'.sses.json')) as f:
            these_sses = json.load(f)[domain]['secondary_structure_elements']
        if length_thresholds != None:
            selected_indices = find_indices_where(these_sses, lambda s: long_enough(s, length_thresholds))
            these_sses  = [these_sses[i] for i in selected_indices]
        else:
            selected_indices = range(len(these_sses))
        sse_labels.extend((domain + '_' + s['label'] for s in these_sses))
        sse_indices_in_domain.extend(selected_indices)
        these_coordinates = [sse['start_vector'] + sse['end_vector'] for sse in these_sses]
        sse_coordinates.extend(these_coordinates)
        progess_bar.step()
        # lib.log(domain,'SSEs:', these_coordinates, '\n')
    progess_bar.finalize()
    N_sses = len(sse_labels)
    offsets.append(N_sses)
    # lib.log('Offsets:', offsets)
    
    lib.log('N_sses:', N_sses)
    supermatrix = np.zeros((N_sses, N_sses), dtype=float)
    sse_coordinates = np.array(sse_coordinates, dtype=float)
    
    progess_bar = ProgressBar(N_sses*(N_sses+1)/2, title='Computing distance matrices from multiple-alignment coordinates').start()
    for i, j in itertools.combinations_with_replacement(range(N_sses), 2):
        diff = sse_coordinates[i,:] - sse_coordinates[j,:]  # 6-dimensional difference vector between the two SSEs
        x1, y1, z1, x2, y2, z2 = diff
        dist = np.sqrt(x1*x1 + y1*y1 + z1*z1) + np.sqrt(x2*x2 + y2*y2 + z2*z2)
        dx, dy, dz = sse_coordinates[i,0:3] - sse_coordinates[i,3:6]
        size_i = np.sqrt(dx*dx + dy*dy + dz*dz)  # size of SSE i
        dx, dy, dz = sse_coordinates[j,0:3] - sse_coordinates[j,3:6]
        size_j = np.sqrt(dx*dx + dy*dy + dz*dz)  # size of SSE j
        if size_penalty_factor != 0:
            size_difference = abs(size_i - size_j)
            dist += size_penalty_factor * size_difference
        if norm_distance_by_size:
            size = min(size_i, size_j)
            factor = 1.0 / (size + NORM_DISTANCE_MIN_SIZE)  #  + NORM_DISTANCE_MIN_SIZE to avoid division by zero
            dist = factor * dist
        supermatrix[i, j] = dist
        supermatrix[j, i] = dist
        progess_bar.step()
    progess_bar.finalize()

    if protein_similarity_weight != 0:
        progess_bar = ProgressBar(n*(n+1)/2, title='Computing distance matrices from multiple-alignment coordinates - adding protein similarity').start()
        for i, j in itertools.combinations_with_replacement(range(len(samples)), 2):
            ij_matrix = supermatrix[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]]
            col_mins = np.min(np.abs(ij_matrix), 0)
            row_mins = np.min(np.abs(ij_matrix), 1)
            min_distances = np.concatenate((col_mins, row_mins))
            protein_similarity = np.mean(min_distances)
            ij_matrix = ij_matrix + protein_similarity * protein_similarity_weight
            supermatrix[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] = ij_matrix
            supermatrix[offsets[j]:offsets[j+1], offsets[i]:offsets[i+1]] = ij_matrix.transpose()
            progess_bar.step()
        progess_bar.finalize()

    # Print the resulting matrix to a file
    lib.print_matrix(supermatrix, path.join(directory, 'distance_matrix.tsv'), sse_labels, sse_labels)
    lib.print_matrix(sse_coordinates, path.join(directory, 'sse_coordinates.tsv'), sse_labels, ['x1', 'y1', 'z1', 'x2', 'y2', 'z2'])

def make_precedence_matrix(names):
    n = len(names)
    precedence = np.zeros((n, n), dtype=bool)
    pdbs, offsets = get_offsets(names, key=lambda name: name.split('_')[0])
    for p in range(len(pdbs)):
        for i, j in itertools.combinations(range(offsets[p], offsets[p+1]), 2):
            precedence[i, j] = True
    return precedence

def write_clustered_sses(directory, original_labels, new_labels):
    domains, offsets = get_offsets(original_labels, key=lambda name: name.split('_')[0])
    n_clusts = max(new_labels)+1
    lengths = np.zeros((n_clusts, len(domains)), dtype=int)
    row_names = ['']*n_clusts
    types = {}
    for d, domain in enumerate(domains):
        with open(path.join(directory, domain+'.sses.json')) as f:
            sses = json.load(f)[domain]['secondary_structure_elements']
        sse_lookup = create_lookup((sse['label'] for sse in sses))
        new_sses = []
        for i in range(offsets[d], offsets[d+1]):
            orig_label = original_labels[i].split('_')[1]
            new_label = new_labels[i]
            sse = sses[sse_lookup[orig_label]]
            typ = two_class_type(sse)
            sse['label'] = typ + str(new_label)
            row_names[new_label] = sse['label']
            sse['color'] = hash_color(new_label)
            new_sses.append(sse)
            lengths[new_label, d] = length(sse)
            if new_label not in types:
                types[new_label] = typ
            elif types[new_label] != typ:
                sys.stderr.write('WARNING: Conflict in types in cluster ' + str(new_label) + '\n')
        output_json = { domain: { 'secondary_structure_elements': new_sses } }
        with open(path.join(directory, domain+'-clust.sses.json'), 'w') as g:
            json.dump(output_json, g)
    lib.print_matrix(lengths, path.join(directory, 'lengths.tsv'), row_names=row_names, col_names=domains)

def write_consensus(directory, cluster_members, type_vector):
    sse_coords, *_ = lib.read_matrix(path.join(directory, 'sse_coordinates.tsv'))
    sses = []
    for cluster, members in enumerate(cluster_members):
        these_coords = lib.submatrix_int_indexing(sse_coords, members, range(6))
        consensus = np.mean(these_coords, 0)
        start_vector = list(consensus[0:3])
        end_vector = list(consensus[3:6])
        typ = ('H' if type_vector[members[0]] == 1 else 'E')
        label = typ + str(cluster)
        color = hash_color(cluster)
        sse = {'label': label, 'type': typ, 'found_in': len(members), 'start_vector': start_vector, 'end_vector': end_vector, 'color': color}
        sses.append(sse)
    with open(path.join(directory, 'consensus.sses.json'), 'w') as w:
        json.dump({'consensus': {'secondary_structure_elements': sses}}, w)

class AcyclicClustering:
    def __init__(self, aggregate_function=lib_clustering.average_linkage_aggregate_function):
        # aggregate_function - how to compute a value (of distance) for the cluster joined from two clusters value1,size1,value2,size2 => result_value
        self.aggregate_function = aggregate_function

    def fit(self, distance_matrix, precedence_matrix, type_vector=None):
        # only samples with the same type can be linked
        n = distance_matrix.shape[0] # number of samples = number of leaves
        m = 2*n - 1 # max. possible number of all nodes
        if distance_matrix.shape[1] != n:
            raise Exception('distance_matrix must be a square matrix')
        if precedence_matrix.shape != (n, n):
            raise Exception('precedence_matrix must have the same size as distance matrix')
        curr_n_nodes = n
        leader = list(range(n))
        members = [[i] for i in range(n)]
        children = np.full((m, 2), -1) # matrix of children pairs, -1 = no child
        self.children = children
        self.distances = np.full(m, 0.0) # distances between the children of each internal node
        active_nodes = set(range(n))
        D = distance_matrix.copy() # working version of distance matrix
        P = precedence_matrix.copy() # working version of precedence matrix
        T = type_vector if type_vector is not None else np.zeros(n, dtype=int) # type vector

        def can_join(ij):
            i, j = ij
            return i in active_nodes and j in active_nodes and T[leader[i]] == T[leader[j]] and not P[leader[i], leader[j]] and not P[leader[j], leader[i]]

        # distance_queue = lib.PriorityQueue(( ((i, j), D[i, j]) for (i, j) in itertools.combinations(active_nodes, 2) if can_join((i, j)) ))
        distance_queue = lib.PriorityQueue(( (D[i, j], (i, j)) for (i, j) in itertools.combinations(active_nodes, 2) if can_join((i, j)) ))
        
        #print('D:', D.shape, 'P:', P.shape)
        with lib.ProgressBar(n-1, title='Clustering ' + str(n) + ' samples') as progress_bar:
            while len(active_nodes) >= 2:
                best_pair = distance_queue.pop_min_which(can_join)
                if best_pair == None:
                    break # no joinable pairs, algorithm has converged
                distance, (p, q) = best_pair
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
                active_leaders = [leader[i] for i in active_nodes]
                to_p = [l for l in active_leaders if P[l, p_leader] and not P[l, q_leader]] + [p_leader]
                to_q = [l for l in active_leaders if P[l, q_leader] and not P[l, p_leader]] + [q_leader]
                from_p = [l for l in active_leaders if P[p_leader, l] and not P[q_leader, l]] + [p_leader]
                from_q = [l for l in active_leaders if P[q_leader, l] and not P[p_leader, l]] + [q_leader]
                for i in to_p:
                    for j in from_q:
                        P[i, j] = True
                for i in to_q:
                    for j in from_p:
                        P[i, j] = True
                # Update distance matrix (calculate distances from the new cluster)
                for i in active_nodes:
                    if i != new:
                        i_leader = leader[i]
                        D[new_leader, i_leader] = self.aggregate_function(D[p_leader, i_leader], len(members[p]), D[q_leader, i_leader], len(members[q]))
                        D[i_leader, new_leader] = D[new_leader, i_leader]
                        if can_join((i,new)):
                            distance_queue.add(D[new_leader, i_leader], (i, new))
                progress_bar.step()

        self.n_clusters = len(active_nodes)
        sorted_active_nodes = lib.sort_dag(active_nodes, lambda i, j: P[leader[i], leader[j]])
        self.final_members = [members[a] for a in sorted_active_nodes]
        sorted_active_leaders = [leader[i] for i in sorted_active_nodes]
        self.cluster_distance_matrix = lib.submatrix_int_indexing(D, sorted_active_leaders, sorted_active_leaders)
        self.labels = np.zeros(n, dtype=int)
        for label in range(self.n_clusters):
            self.labels[self.final_members[label]] = label
        return

################################################################################

# Read the list of domains
with open(path.join(args.directory, 'sample.json')) as f:
    samples = json.load(f)
# lib.log('Domains:', samples)
lib.log(len(samples), 'domains')

# Try some stuff
# protein_similarity_matrix, _, _ = lib.read_matrix(path.join(args.directory, 'mapsci_score_matrix.tsv'))
# protein_similarity_matrix, _, _ = lib.read_matrix(path.join(args.directory, 'rmsds.tsv')); #protein_similarity_matrix = protein_similarity_matrix**2
protein_similarity_matrix, _, _ = lib.read_matrix(path.join(args.directory, 'q_scores.tsv')); protein_similarity_matrix = protein_similarity_matrix**(-1) # protein_similarity_matrix = -np.log(protein_similarity_matrix)**2
# protein_similarity_matrix, _, _ = lib.read_matrix(path.join(args.directory, 'R_scores.tsv'))

mean_similarity = np.mean(protein_similarity_matrix.flatten())
protein_similarity_matrix /= mean_similarity
m = protein_similarity_matrix.shape[0]
ac_prot = AcyclicClustering(aggregate_function=lib_clustering.average_linkage_aggregate_function)
ac_prot.fit(protein_similarity_matrix, np.zeros_like(protein_similarity_matrix))

n = protein_similarity_matrix.shape[0]
last = 2*n - ac_prot.n_clusters - 1
start = 2*n - 2*ac_prot.n_clusters
stop = 2*n - ac_prot.n_clusters
print('Children:', ac_prot.children.shape, 'Clusters:', ac_prot.n_clusters)
print(last, start, stop)
root1, root2 = ac_prot.children[last,:]
branch1 = lib_clustering.children_to_list(ac_prot.children, root=root1, node_names=[t[1] for t in samples])
branch2 = lib_clustering.children_to_list(ac_prot.children, root=root2, node_names=[t[1] for t in samples])
branch1indices = lib_clustering.children_to_list(ac_prot.children, root=root1)
branch2indices = lib_clustering.children_to_list(ac_prot.children, root=root2)
print('Branch 1:')
print(*branch1)
print('Branch 2:')
print(*branch2)
with open(path.join(args.directory, 'sample_branch1.json'), 'w') as w:
    json.dump([ samples[i] for i in branch1indices], w, indent=4)
with open(path.join(args.directory, 'sample_branch2.json'), 'w') as w:
    json.dump([ samples[i] for i in branch2indices], w, indent=4)

newi = lib_clustering.children_to_newick(ac_prot.children, [stop-1], distances=ac_prot.distances, node_names=[t[1] for t in samples])
print(newi)
t = Tree(newi)
print('n_leaves:', len([*t.iter_leaves()]))
print(*ac_prot.distances) 
print('cluster_sizes:', *( len([*x.iter_leaves()]) for x in t.get_children() ))
t.show()
