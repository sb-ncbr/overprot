# Calculate some overview of CATH number of families, domains etc.

import json
import os
from os import path
import argparse
from collections import defaultdict
from matplotlib import pyplot as plt

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('cath_domain_list', help='CATH domain list file (like ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt)', type=str)
args = parser.parse_args()

cath_domain_list_file = args.cath_domain_list

################################################################################

# ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/
# 
# Column 1:  CATH domain name (seven characters)
# Column 2:  Class number
# Column 3:  Architecture number
# Column 4:  Topology number
# Column 5:  Homologous superfamily number
# Column 6:  S35 sequence cluster number
# Column 7:  S60 sequence cluster number
# Column 8:  S95 sequence cluster number
# Column 9:  S100 sequence cluster number
# Column 10: S100 sequence count number
# Column 11: Domain length
# Column 12: Structure resolution (Angstroms)
#            (999.000 for NMR structures and 1000.000 for obsolete PDB entries)
# 
# C.A.T.H.S.O.L.I.D
# C - Class
# A - Architecture
# T - Topology
# H - Homologous Superfamily
# S - Sequence Family (S35)
# O - Orthogous Seqeuce Family (S60)
# L - 'Like' Sequence Family (S95)
# I - Identical (S100)
# D - Domain (S100 count)

family2domains = defaultdict(set)
family2idfams = defaultdict(set)

with open(cath_domain_list_file) as f:
    for line in f:
        if not line.startswith('#'):
            domain, c, a, t, h, s, o, l, i, i_size, length, resolution = line.split()
            family = f'{c}.{a}.{t}.{h}'
            idfam = f'{s}.{o}.{l}.{i}'
            family2domains[family].add(domain)
            family2idfams[family].add(idfam)

families = list(family2domains.keys())
ns_idfams = [len(idfams) for fam, idfams in family2idfams.items()]
ns_domains = [len(domains) for fam, domains in family2domains.items()]
classes = [int(fam.split('.')[0]) for fam in families]

n_families = len(families)
# print(families)
# print(ns_idfams)
# print(ns_domains)

plt.scatter(x=ns_domains, y=ns_idfams, c=classes)
plt.xlabel('#domains in the family')
plt.ylabel('#identical families (S100) in the family')
plt.xscale('log')
plt.yscale('log')
for family, n_domains, n_idfams in zip(families, ns_domains, ns_idfams):
    if family == '1.10.630.10':
        print(family, n_domains, n_idfams)
    # plt.annotate(family, (n_domains, n_idfams))
plt.show()
