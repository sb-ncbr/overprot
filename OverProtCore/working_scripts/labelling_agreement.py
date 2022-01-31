
import argparse
from pathlib import Path
import numpy as np
from libs import lib_acyclic_clustering

parser = argparse.ArgumentParser()
parser.add_argument('labels1', help='File with labels (one integer per line)', type=Path)
parser.add_argument('labels2', help='File with labels (one integer per line)', type=Path)
parser.add_argument('--strict', help='Do not match classes, require exactly the same label', action='store_true')
parser.add_argument('--remove_unclassified', help='Do not match classes, require exactly the same label', action='store_true')
args = parser.parse_args()


with open(args.labels1) as r:
    labels1 = np.array([ int(l) for l in r.read().split() ])
with open(args.labels2) as r:
    labels2 = np.array([ int(l) for l in r.read().split() ])

agreement = lib_acyclic_clustering.labelling_agreement(labels1, labels2, allow_matching=not args.strict, include_both_unclassified=not args.remove_unclassified)
print(agreement)