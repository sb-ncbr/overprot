
// Implementation of the dynamic programming algorithm for global alignment.


#include <iostream>

#include "global.h"
#include "alignment.h"
#include "matrix.h"


// Allocates space for the tables used in the global alignment.
// For efficiency reasons, the space for these tables is not released,
// although they will grow in size as needed and will occupy the max 
// allocated space throughout the use of the algorithm (however they 
// will only expose the section needed by the current alignment)
//
GlobalAlign::GlobalAlign() : D(100, 100), V(100, 100)
{
}


// resizes the tables used by the algorithm based on 
// the lengths of the "sequences" being aligned
void GlobalAlign::resize(int rows, int cols)
{
  D.resize(rows, cols);
  V.resize(rows, cols);
}


// the standard intialization with initial gap penalties
void GlobalAlign::init()
{
  V(0, 0) = 0;
  D(0, 0) = 0;
    
  for (int i = 1; i < V.rows(); i++) {
    V(i, 0) = V(i-1, 0) + Params::GAP_PENALTY;
    D(i, 0) = VERT;
  }
    
  for (int j = 1; j < V.cols(); j++) {
    V(0, j) = V(0, j-1) + Params::GAP_PENALTY;
    D(0, j) = HORI;
  }
}


// traverses the direction matrix to recreate the aliagnment
void GlobalAlign::build(Alignment* align)
{
  int m = D.rows();
  int n = D.cols();

  // longest possible size for alignment is rows+cols-1
  align->resize(m+n);
    
  int i = m - 1;
  int j = n - 1;
  int count = 0;

  int matches = 0;

  while ( (i > 0) || (j > 0) ) {
      
    if (D(i, j) == DIAG) {
      i--;
      j--;
      align->first[count]   = i;
      align->second[count]  = j;

      ++matches;
    }
    else if (D(i, j) == VERT) {
      i--;
      align->first[count]  = i;
      align->second[count] = GAP_INDEX;
    }
    else if (D(i, j) == HORI) {
      j--;
      align->first[count]  = GAP_INDEX;
      align->second[count] = j;
    }
    else {
      std::cout << "global: invalid direction" << std::endl;
    }
      
    count++;
  }

  //reverse the alignment correspondence
  for (int k = 0; k < count/2; ++k) {
    int temp = align->first[k];
    align->first[k] = align->first[count-1-k];
    align->first[count-1-k] = temp;
      
    temp = align->second[k];
    align->second[k] = align->second[count-1-k];
    align->second[count-1-k] = temp;
  }
    
  align->size = count;
  align->score = V(m-1, n-1);
  align->matches = matches;
}
