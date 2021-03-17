

// Implementation of the algorithm for pairwise structure alignment in
//
//   Ye et al., "Pairwise Protein Structure Alignment ...",
//              JBCB, 2 (4): 699--718 (2004).


#include <assert.h>
#include <cfloat>


#include "pairwise.h"
#include "alignment.h"
#include "runs.h"
#include "protein.h"
#include "point.h"
#include "matrix.h"
#include "msvd.h"



// reserve small default space for runs
PairAlign::PairAlign() : runs(20)
{
}


// does pairwise structure alignment of the given algorithms; a pointer to
// a previously allocated alignment must be provided as the last parameter,
// which will be modified to contain the final alignment
//
// for details of the code see Ye at al., JBCB 2 (4) 2004, Section 3, p. 5
//
void PairAlign::align(Protein* prot1, Protein* prot2, Alignment* ali) 
{
  // Step 3.1, 3.2: Semiglobal alignment of the angle triplets
  //  semiglobal.align(prot1, prot2, ali);
  semiglobal.align(prot1->angles, prot2->angles, AngleTriple::distance, ali);

  int count = 0;
  double diff = DBL_MAX;
  double cur_score = DBL_MAX;

  int extend = 3;

  do
  {
    // Step 3.3: Identifying the aligned sections (contiguous runs), followed by
    //           identification of consistent runs (similar rotation matrices)
    runs.buildContiguous(ali, prot1, prot2, extend); 
    vector<int> clique = cf.consistent(runs, prot1, prot2); 
    
    extend = 0;

    // use the points in consistent run to find optimal
    // transformations for superposing prot1 onto prot2
    if (runs.size() != 0) {
      superpose(prot1, prot2, clique, runs);
    }


    // Step 3.4: Refine the alignment based on actual coordinates (not angles)
    global.align(prot1->atoms, prot2->atoms, Point::squaredDistance, ali);

    // Compute current score to determine convergence
    double rmsd = ali->computeRMSD(prot1, prot2);
    diff = fabs(rmsd - cur_score);
    cur_score = rmsd;
    count++;
  }
  while ( (diff > 0.1) && (count < Params::MAX_ITERS) );
}


// superposes prot1 onto prot2 via the transformation matrices
// extracted from the given consistent contiguous runs
void PairAlign::superpose(Protein* prot1, Protein* prot2,
			  const vector<int>& clique,
			  const RunsList& runs)
{
  vector<Point> A;
  vector<Point> B;
  
  int n = 0;
  for (int i = 0; i < clique.size(); i++) {
    int node = clique[i];
    for (int j = 0; j < runs[node].size; j++) {
      A.push_back( prot1->atoms[runs[node].start1 + j] );
      B.push_back( prot2->atoms[runs[node].start2 + j] );
      
      n++;
    } 
  }

  // find the optimal transformations for the matched atoms via SVD
  Matrix<double> rot;
  Point trans;
  
  MSVD::svd(A, B, 0, n-1, 0, n-1, rot, trans);

  prot1->orient(rot, trans);
}
