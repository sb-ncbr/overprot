

#include <cmath>
#include <assert.h>

#include <vector>
using std::vector;


#include "zscore.h"
#include "alignment.h"
#include "protein.h"
#include "point.h"


// given an alignment and core diamter computes the core size and core RMSD;
// core is defined as the set of conserved columns (i.e. no gaps) such that
// every pair of atoms is within the given core diamater (e.g. 4A)
//
void compute_core(const vector<Alignment*>& MA, 
		  const vector<Protein*>&   M, 
		  double  coreDiam,
		  double& coreRmsd,
		  int&    coreCount)
{
  int count = 0;
  double rmsd = 0;

  // for each column (m) in the multiple alignment
  for (int m = 0; m < MA[0]->size; ++m) {  

    // look for gaps in the current alignment column
    bool gaps = false;
    for (int i = 0; i < M.size() && !gaps; ++i) {
      int index = MA[i]->first[m];
      if (index == GAP_INDEX ) {
	gaps = true;
      }  
    }
    if (gaps) {
      continue;
    }


    // check whether every pair of atoms in the current
    // column is with the specified core diameter
    bool isCore = true;
    for (int i = 1; i < M.size() && isCore; ++i) {
      for (int j = 0; j < i && isCore; ++j) {
	int p = MA[i]->first[m];
	int q = MA[j]->first[m];
	
	Point atom1 = M[i]->atoms[p];
	Point atom2 = M[j]->atoms[q];

	double dist = Point::distance(atom1, atom2);
	if (dist > coreDiam) {
	  isCore = false;
	}
      }
    }
    if (!isCore) {
      continue;
    }

    
    // find centroid (cx, cy, cz) of current column
    Point cen(0, 0, 0);
    for (int i = 0; i < M.size(); ++i) {
      int index = MA[i]->first[m];
      Point atom = M[i]->atoms[index];
      cen += atom;
    }  
    cen /= M.size();
    
    // calculate the squared distance to centroid
    // for all atoms in the current column
    double dist = 0;
    for (int i = 0; i < M.size(); ++i) {
      int index = MA[i]->first[m];
      Point atom = M[i]->atoms[index];
      
      dist += Point::squaredDistance(cen, atom);
    }
    
    rmsd += sqrt(dist/M.size());
    ++count;
  }

  rmsd = (count == 0) ? -1 : rmsd / count;
  
  coreCount = count;
  coreRmsd = rmsd;
}


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
		    double& outLnP)
{
  assert(norm != 0);
  
  double psi = (double(core) / norm) * 100;

  //     EV fitting, using N>70
  double am = 747.29 * pow(norm, -0.7971);
  double as = 124.99 * pow(norm, -0.6882);
  double zscore = (psi-am)/as;
  
  //     Extreme Value approach
  double znew = 0.730*(1.2825755*zscore + 0.5772);
  double ev = 1.0-exp(-exp(-znew));
  
  //     due to numerical errors, the e-value is cutoff at 2.650e-14;
  double lne = 0;
  if (ev < 2.650E-14) {
    lne = zscore;
  }                                                           
  else {
    lne = -log(ev);
  }

  outZ = zscore;
  outLnP = lne;
}
