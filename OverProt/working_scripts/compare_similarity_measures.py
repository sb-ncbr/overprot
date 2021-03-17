
import json
import os
from os import path
import sys
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import argparse
from libs import lib
from libs import lib_acyclic_clustering_simple

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('directory', help='Directory with sample.json and structure files', type=str)
args = parser.parse_args()
directory = args.directory

QUASI_INFINITY = 1e6
CYTOS_BINARY = 'dotnet SecStrAnnotator/SecStrAnnotator.dll'

################################################################################

def sse_distance(coords1, coords2):
        diff = coords1 - coords2
        x1, y1, z1, x2, y2, z2 = diff
        dist = np.sqrt(x1*x1 + y1*y1 + z1*z1) + np.sqrt(x2*x2 + y2*y2 + z2*z2)
        # dx, dy, dz = coords1[0:3] - coords1[3:6]
        # size_i = np.sqrt(dx*dx + dy*dy + dz*dz)  # size of SSE i
        # dx, dy, dz = coords2[0:3] - coords2[3:6]
        # size_j = np.sqrt(dx*dx + dy*dy + dz*dz)  # size of SSE j
        # if SIZE_PENdomainsCTOR != 0:
        # 	size_difference = abs(size_i - size_j)
        # 	dist += SIZE_PENALTY_FACTOR * size_difference
        # if NORM_DISTANCE_BY_SIZE:
        # 	size = min(size_i, size_j)
        # 	factor = 1.0 / (size + NORM_DISTANCE_MIN_SIZE)  #  + NORM_DISTANCE_MIN_SIZE to avoid division by zero
        # 	dist = factor * dist
        return dist

def line_segment_lengths(start_coords, end_coords):
    '''coords - matrix n*3'''
    return np.sqrt(np.sum((end_coords - start_coords)**2, axis=1))

def score_matrix_with_secstrannotator(samples, directory, append_outputs=True, cealign=True):
    score_matrix_file = path.join(directory, 'secstrannot_score' + ('_cealign' if cealign else '') + '.tsv')
    if path.isfile(score_matrix_file):
        score, _, _ = read_matrix(score_matrix_file)
        return score
    stdout_file, stderr_file = get_out_err_files(directory, append=append_outputs)
    n = len(samples)
    domains = [ domain for pdb, domain, chain, rang in samples ]
    compute_ssa(samples, directory, skip_if_exists=True)
    align_method = '--align cealign' if cealign else '--align none'
    score = np.zeros((n, n), dtype=float)
    with lib.ProgressBar(n*(n-1)/2, title='Computing distance matrices') as progess_bar:
        for i, j in itertools.combinations_with_replacement(range(len(samples)), 2):
            pi, di, ci, ri = samples[i]
            pj, dj, cj, rj = samples[j]
            run_command(CYTOS_BINARY, align_method, '--ssa file', directory, di + ',' + ci + ',' + ri , dj + ',' + cj + ',' + rj, stdout=stdout_file, stderr=stderr_file, appendout=True, appenderr=True)
            # run_command('mv', path.join(directory, 'metric_matrix.tsv'), path.join(directory, 'matrix-'+di+'-'+dj+'.tsv'))
            run_command('rm', '-f', path.join(directory, 'alignment-'+di+'-'+dj+'.json'))
            run_command('rm', '-f', path.join(directory, dj+'-detected.sses.json'))
            #run_command('rm', '-f', path.join(directory, dj+'-aligned.pdb'))
            with open(path.join(directory, dj+'-annotated.sses.json')) as r:
                s = json.load(r)[dj]['total_metric_value']
            score[i, j] = s
            score[j, i] = s
            progess_bar.step()
        run_command('rm', '-f', path.join(directory, 'template-smooth.pdb'))
    lib.print_matrix(score, score_matrix_file, domains, domains)
    return score

def average_min_dist(distance_matrix, offsets, average_function=np.mean, symmetric=False):
    n = len(offsets) - 1
    result = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            submatrix = distance_matrix[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]]
            minima = submatrix.min(axis=1)
            result[i, j] = average_function(minima)
    if symmetric:
        result = 0.5 * (result + result.transpose())
    return result

def ranks(array):
    order = array.argsort()
    ranks = order.argsort()
    return ranks

def plot_flat(X, Y):
    plt.plot(X.flatten(), Y.flatten(), '.')
    r = stats.spearmanr(X.flatten(), Y.flatten()).correlation
    plt.title(f'Spearman: {r:.3f}')
    plt.show()

def plot_ranks(X, Y):
    plt.plot(ranks(X.flatten()), ranks(Y.flatten()), '.')
    r = stats.spearmanr(X.flatten(), Y.flatten()).correlation
    plt.title(f'Spearman: {r:.3f}')
    plt.show()

################################################################################

# Read the list of domains
with open(path.join(directory, 'sample.json')) as f:
    samples = json.load(f)
domains = [ domain for pdb, domain, chain, ranges in samples ]
n_domains = len(domains)
# lib.log('Domains:', samples)
lib.log(n_domains, 'domains')

# Try some stuff
mapsci_score, _, _ = lib.read_matrix(path.join(args.directory, 'pdb_mapsci', 'mapsci_score_matrix.tsv'))
q_score_mapsci, _, _ = lib.read_matrix(path.join(args.directory,'pdb_mapsci', 'q_score_matrix.tsv'))

rmsd, _, _ = lib.read_matrix(path.join(args.directory, 'rmsds.tsv'))
# rmscur, _, _ = lib.read_matrix(path.join(args.directory, 'rmscurs.tsv'))
q_score, _, _ = lib.read_matrix(path.join(args.directory, 'q_scores.tsv'))
# q_score_cur, _, _ = lib.read_matrix(path.join(args.directory, 'q_scores_cur.tsv'))
# r_score, _, _ = lib.read_matrix(path.join(args.directory, 'R_scores.tsv'))

offsets, sses, coordinates, distance, start_distance, end_distance, type_vector = lib_acyclic_clustering_simple.read_sses_simple(samples, path.join(directory, 'cif_cealign'))
segment_lengths = line_segment_lengths(coordinates[:, 0:3], coordinates[:, 3:6])
mean_min = average_min_dist(distance, offsets, symmetric=True)
median_min = average_min_dist(distance, offsets, average_function=np.median, symmetric=True)

mean_min_se = 0.5 * (average_min_dist(start_distance, offsets, symmetric=True) + average_min_dist(end_distance, offsets, symmetric=True))
median_min_se = 0.5 * (average_min_dist(start_distance, offsets, average_function=np.median, symmetric=True) + average_min_dist(end_distance, offsets, average_function=np.median, symmetric=True))

K0 = 30
scores = K0 - distance
scores[scores < 0] = 0
type_mismatch = lib.each_to_each(np.not_equal, type_vector)
scores[type_mismatch] = 0
scores_weighted = scores * lib.each_to_each(np.minimum, segment_lengths)

# scores_weighted_prod = scores * lib.each_to_each(np.multiply, segment_lengths)
# scores_weighted_geom = scores * np.sqrt(lib.each_to_each(np.multiply, segment_lengths))
# scores_weighted_sum = scores * lib.each_to_each(np.add, segment_lengths)

dynprog_scores = lib_acyclic_clustering_simple.dynprog_total_scores_each_to_each(scores_weighted, offsets)

fup = lib_acyclic_clustering_simple.dynprog_fuckup_indices_each_to_each(scores_weighted, offsets)
lib.print_matrix(scores_weighted[offsets[0]:offsets[1], offsets[1]:offsets[2]], path.join(args.directory, 'score_matrix.tsv'))
lib.print_matrix(fup[offsets[0]:offsets[1], offsets[1]:offsets[2]], path.join(args.directory, 'fuckup_matrix.tsv'))

lib.print_matrix(dynprog_scores, path.join(args.directory, 'dynprog_scores.tsv'), row_names=domains, col_names=domains)

# secstrannot_score = score_matrix_with_secstrannotator(samples, path.join(directory, 'cif_cealign'), cealign=False)
# secstrannot_score_cealign = score_matrix_with_secstrannotator(samples, path.join(directory, 'cif_cealign'), cealign=True)

plot_flat(q_score, rmsd)
plot_flat(q_score, mean_min)
plot_flat(q_score, dynprog_scores)

# plot_flat(q_score, mean_min)
# plot_flat(q_score, mean_min_se)
# plot_flat(q_score, median_min)
# plot_flat(q_score, median_min_se)

# plot_flat(rmsd, mean_min)
# plot_flat(rmsd, mean_min_se)
# plot_flat(rmsd, median_min)
# plot_flat(rmsd, median_min_se)