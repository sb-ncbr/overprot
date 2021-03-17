import json
import os
from os import path
import sys
import numpy as np

HOW_TO_RUN = 'pymol -qcr cealign_all_to_all.py -- FILES_OR_DIR'


def print_usage():
	print('Usage: ' + HOW_TO_RUN + '\n')
	print('FILES_OR_DIR can be:\n\
	1. Multiple CIF files, then outputs will be in "outputs/".\n\
	2. A directory with "sample.json" and CIF files in "cif_cealign/", then outputs will be in this directory.')
	exit(1)

if len(sys.argv) < 2:
	print_usage()

try:
	from pymol import cmd
	try:
		cmd.cealign('a', 'a')
	except pymol.CmdException:
		pass
except:
	print_usage()
cmd.set('cif_use_auth', False)

####################################################################################

def print_matrix(matrix, filename, row_names=None, col_names=None, sep='\t', comment_symbol='#'):
	if matrix.dtype==bool:
		str_ = lambda x: '1' if x else '0'
	elif matrix.dtype==float:
		str_ = '{:g}'.format
	else:
		str_ = str
	m, n = matrix.shape
	with open(filename, 'w') as g:
		g.write(comment_symbol + 'dtype='+str(matrix.dtype) + '\n')
		g.write(comment_symbol + 'shape='+str(m)+','+str(n) + '\n')
		if row_names!=None and col_names!=None:
			g.write(sep)
		if col_names!=None:
			g.write(sep.join(col_names) + '\n')
		for i in range(m):
			if row_names!=None:
				g.write(row_names[i] + sep)
			g.write(sep.join((str_(x) for x in matrix[i,:])) + '\n')

def random_pair_values(matrix, n_pairs=100):
	m, n = matrix.shape
	if m != n:
		raise
	result = []
	for i in range(n_pairs):
		p, q = np.random.choice(range(n), 2)
		result.append(matrix[p, q])
	return sorted(result)

class ProgressBar:
	def __init__(self, n_steps, width=100, title=''):
		self.n_steps = n_steps # expected number of steps
		self.width = width
		self.title = title
		self.done = 0 # number of completed steps
		self.shown = 0 # number of shown symbols
	def start(self):
		print('| ' + self.title + ' |')
		return self
	def step(self, n_steps=1):
		self.done = min(self.done + n_steps, self.n_steps)
		new_shown = int(self.width * self.done / self.n_steps)
		if new_shown > self.shown:
			print('|%4d %%' % int(100*new_shown/self.width))
		self.shown = new_shown
	def finalize(self):
		self.step(self.n_steps - self.done)


####################################################################################

filenames = sys.argv[1:]
if len(filenames) == 1 and path.isdir(filenames[0]):  # single argument is directory with sample.json and structures in cif_cealign
	directory = filenames[0]
	with open(path.join(directory, 'sample.json')) as r:
		sample = json.load(r)
	basenames = [ str(domain) + '.cif' for pdb, domain, chain, ranges in sample ]
	filenames = [ path.join(directory, 'cif_cealign', bn) for bn in basenames ]
else:  # arguments are structure files
	directory = path.join(os.getcwd(), 'outputs')
	basenames = [path.basename(fn) for fn in filenames]

for filename, basename in zip(filenames, basenames):
	cmd.load(filename, basename)  # Looks like it's not faster than loading each file every time it's needed (at least for cyps_50)

n = len(basenames)
R0 = 3.0

rmsds = np.zeros((n, n))
rmscurs = np.zeros((n, n))
ali_lengths = np.zeros((n, n))
progress_bar = ProgressBar(n*(n+1)/2, width=20, title='Running cealigns').start()
for i in range(n):
	# cmd.load(filenames[i], basenames[i])
	for j in range(i, n):
		# if i != j:
		# 	cmd.load(filenames[j], basenames[j])
		# print(basenames[i], basenames[j])
		aln_result = cmd.cealign(basenames[i], basenames[j], transform=0, object='aln')
		rmscur = cmd.rms_cur('aln & name CA & ' + basenames[i], 'aln & name CA &' + basenames[j], matchmaker=-1)
		cmd.delete('aln')
		if rmscur > 10:
			print(basenames[i], basenames[j], rmscur)
		rmsd = aln_result['RMSD']
		length = aln_result['alignment_length']
		rmsds[i, j] = rmsd
		rmsds[j, i] = rmsd
		rmscurs[i, j] = rmscur
		rmscurs[j, i] = rmscur
		ali_lengths[i, j] = length
		ali_lengths[j, i] = length
		# if i != j:
		# 	cmd.delete(basenames[j])
		progress_bar.step()
	# cmd.delete(basenames[i])
progress_bar.finalize()
		
lengths = ali_lengths.diagonal()
q_scores = np.zeros((n, n))
q_scores_cur = np.zeros((n, n))
for i in range(n):
	for j in range(i, n):
		q_score = ali_lengths[i, j]**2 / (lengths[i] * lengths[j] * (1.0 + (rmsds[i, j]/R0)**2))
		q_score_cur = ali_lengths[i, j]**2 / (lengths[i] * lengths[j] * (1.0 + (rmscurs[i, j]/R0)**2))
		# q_score = 1.0 / (1.0 + (rmsds[i, j]/R0)**2)
		# q_score = ali_lengths[i, j]**2 / (lengths[i] * lengths[j])
		q_scores[i, j] = q_score
		q_scores[j, i] = q_score
		q_scores_cur[i, j] = q_score_cur
		q_scores_cur[j, i] = q_score_cur

# rand_pairs = random_pair_values(q_scores, n_pairs=10000)
# from matplotlib import pyplot as plt
# plt.hist(rand_pairs)
# plt.show()

try:
	os.mkdir('outputs')
except:
	pass
print_matrix(rmsds, path.join(directory, 'rmsds.tsv'), basenames, basenames)
print_matrix(rmscurs, path.join(directory, 'rmscurs.tsv'), basenames, basenames)
print_matrix(ali_lengths, path.join(directory, 'ali_lengths.tsv'), basenames, basenames)
print_matrix(q_scores, path.join(directory, 'q_scores.tsv'), basenames, basenames)
print_matrix(q_scores_cur, path.join(directory, 'q_scores_cur.tsv'), basenames, basenames)

os.system('echo Q-score > ' + path.join(directory, 'all_matrices.tsv'))
os.system('sed /^#/d ' + path.join(directory, 'q_scores.tsv') + ' >> ' + path.join(directory, 'all_matrices.tsv'))
os.system('echo >> ' + path.join(directory, 'all_matrices.tsv'))
os.system('echo RMSD >> ' + path.join(directory, 'all_matrices.tsv'))
os.system('sed /^#/d ' + path.join(directory, 'rmsds.tsv') + ' >> ' + path.join(directory, 'all_matrices.tsv'))
os.system('echo RMS cur >> ' + path.join(directory, 'all_matrices.tsv'))
os.system('sed /^#/d ' + path.join(directory, 'rmscurs.tsv') + ' >> ' + path.join(directory, 'all_matrices.tsv'))
os.system('echo >> ' + path.join(directory, 'all_matrices.tsv'))
os.system('echo Alignment length >> ' + path.join(directory, 'all_matrices.tsv'))
os.system('sed /^#/d ' + path.join(directory, 'ali_lengths.tsv') + ' >> ' + path.join(directory, 'all_matrices.tsv'))

cmd.quit()