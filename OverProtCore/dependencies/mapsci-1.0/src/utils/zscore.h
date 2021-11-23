

#ifndef __ZSCORE_H__
#define __ZSCORE_H__


// Functions for computing the z-score and statistics about
// the multiple alignment core as described in:
//
//   Lupyan at al. "A new progressive-iterative algorithm for multiple
//                  structure alignment", Bioinformatics, 21, 3255-3263, 2005.


#include <vector>
using std::vector;


struct Alignment;
class Protein;



// given an alignment and core diameter computes the core size and core RMSD;
// core is defined as the set of conserved columns (i.e. no gaps) such that
// every pair of atoms is within the given core diamater (e.g. 4A)
//
void compute_core(const vector<Alignment*>& MA, 
		  const vector<Protein*>& M, 
		  double  core_diam,
		  double& outRmsd,
		  int&    outCore);


// Computes the z-score and e-value. This code is taken from
// the source of MAMMOTH. The original is in Fortran and has 
// been converted here to C.
//
// core -- number of atoms in core
// norm -- length of shortest protein
//
void compute_zscore(int core,
		    int norm,
		    double& outZ,
		    double& outLnP);


#endif
