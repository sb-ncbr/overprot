

#ifndef __PAIRWISE_STRUCTURE_H__
#define __PAIRWISE_STRUCTURE_H__


// Implementation of the algorithm for pairwise structure alignment in
//
//   Ye et al., "Pairwise Protein Structure Alignment ...",
//              JBCB, 2 (4): 699--718 (2004).


#include <vector>
using std::vector;


#include "semiglobal.h"
#include "global.h"
#include "runs.h"
#include "clique.h"

struct Alignment;
class Protein;



class PairAlign
{
public:
  PairAlign();


  // does pairwise structure alignment of the given algorithms; a pointer to
  // a previously allocated alignment must be provided as the last parameter,-
  // which will be modified to contain the final alignment
  //
  void align(Protein* prot1, Protein* prot2, Alignment* ali);

private:
  // superposes prot1 onto prot2 via the transformation matrices
  // extracted from the given consistent contiguous runs  
  void superpose(Protein* prot1, Protein* prot2,
		 const vector<int>& clique,
		 const RunsList& runs);

  
private:
  SemiGlobalAlign semiglobal;
  GlobalAlign global;
  CliqueFinder cf;
  RunsList runs;
};


#endif
