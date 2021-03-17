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
import subprocess

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('directory', help='Directory with sample.json and structure files', type=str)
args = parser.parse_args()

QUASI_INFINITY = 1e6
COMMENT_SYMBOL = '#'
CYTOS_BINARY = 'dotnet SecStrAnnotator/SecStrAnnotator.dll'
RE_DTYPE = re.compile('^\s*'+re.escape(COMMENT_SYMBOL)+'\s*dtype\s*=\s*(\w+)\s*$')
RE_SHAPE = re.compile('^\s*'+re.escape(COMMENT_SYMBOL)+'\s*shape\s*=\s*(\w+)\s*,\s*(\w+)\s*$')
RE_CHAR_DIST = re.compile('Characteristic distance R = (\d+\.\d+)')
# RE_CHAR_DIST = re.compile('Characteristic distance R = .+')
	
	#Characteristic distance R = 4.237

################################################################################

def log(*args):
	print(*args)

def run_command(*params, stdin=None, stdout=None, stderr=None, appendout=False, appenderr=False, do_print=False):
	command = ' '.join(params)
	if stdin!=None:
		command = command + ' < ' + stdin
	if stdout!=None:
		command = command + (' 1>> ' if appendout else ' 1> ') + stdout
	if stderr!=None:
		command = command + (' 2>> ' if appenderr else ' 2> ') + stderr
	if do_print:
		print(command)
	return os.system(command)

def get_out_err_files(directory, append=True):
	stdout_file = path.join(directory, 'stdout.txt')
	stderr_file = path.join(directory, 'stderr.txt')
	if not append:
		with open(stdout_file, 'w') as g:
			g.write('')
		with open(stderr_file, 'w') as g:
			g.write('')
	return stdout_file, stderr_file

def compute_characteristic_distances(samples, directory, append_outputs=True):
	stdout_file, stderr_file = get_out_err_files(directory, append=append_outputs)
	n = len(samples)
	R_matrix = np.zeros((n, n))
	run_command('echo', stdout=path.join(directory, 'stderr.txt'))
	progess_bar = ProgressBar(n*(n+1)/2, title='Computing characteristic distance matrix').start()
	for i, j in itertools.combinations_with_replacement(range(len(samples)), 2):
		pi, di, ci, ri = samples[i]
		pj, dj, cj, rj = samples[j]
		run_command('echo', di, dj, stdout=path.join(directory, 'stderr.txt'), appendout=True)
		command = ' '.join((CYTOS_BINARY, '--matching none', directory, di + ',' + ci, dj + ',' + cj, '2>>', path.join(directory, 'stderr.txt')))
		output = subprocess.check_output(command, shell=True).decode()
		run_command('rm', '-f', path.join(directory, dj+'-aligned.cif'))
		# print(output)
		R  = float(RE_CHAR_DIST.findall(output)[0])
		R_matrix[i, j] = R
		R_matrix[j, i] = R
		progess_bar.step()
	# run_command('rm', '-f', path.join(directory, 'template-smooth.pdb'))
	progess_bar.finalize()
	return R_matrix

def print_matrix(matrix, filename, row_names=None, col_names=None, sep='\t'):
	if matrix.dtype==bool:
		str_ = lambda x: '1' if x else '0'
	elif matrix.dtype==float:
		str_ = '{:g}'.format
	else:
		str_ = str
	m, n = matrix.shape
	with open(filename, 'w') as g:
		g.write(COMMENT_SYMBOL + 'dtype='+str(matrix.dtype) + '\n')
		g.write(COMMENT_SYMBOL + 'shape='+str(m)+','+str(n) + '\n')
		if row_names!=None and col_names!=None:
			g.write(sep)
		if col_names!=None:
			g.write(sep.join(col_names) + '\n')
		for i in range(m):
			if row_names!=None:
				g.write(row_names[i] + sep)
			g.write(sep.join((str_(x) for x in matrix[i,:])) + '\n')

class ProgressBar:
	def __init__(self, n_steps, width=100, title='', writer=sys.stdout):
		self.n_steps = n_steps # expected number of steps
		self.width = width
		self.title = (' '+title+' ')[0:min(len(title)+2, width)]
		self.writer = writer
		self.done = 0 # number of completed steps
		self.shown = 0 # number of shown symbols
	def start(self):
		self.writer.write('|' + self.title + '_'*(self.width-len(self.title)) + '|\n')
		self.writer.write('|')
		self.writer.flush()
		return self
	def step(self, n_steps=1):
		self.done = min(self.done + n_steps, self.n_steps)
		new_shown = int(self.width * self.done / self.n_steps)
		self.writer.write('*' * (new_shown-self.shown))
		self.writer.flush()
		self.shown = new_shown
	def finalize(self):
		self.step(self.n_steps - self.done)
		self.writer.write('|\n')
		self.writer.flush()

################################################################################

# Read the list of domains
with open(path.join(args.directory, 'sample.json')) as f:
	samples = json.load(f)
#log('Domains:', samples)
log(len(samples), 'domains')

R_matrix = compute_characteristic_distances(samples, path.join(args.directory, 'cif_cealign'))
print(R_matrix)
domain_names = [d for p, d, c, r in samples]
print_matrix(R_matrix, path.join(args.directory, 'R_scores.tsv'), row_names=domain_names, col_names=domain_names)