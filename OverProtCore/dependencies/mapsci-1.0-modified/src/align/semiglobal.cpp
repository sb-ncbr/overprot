
// Implementation of the dynamic programming algorithm for semiglobal alignment.
// Gaps are assigned an affine penalty g(n) = gap_open + n*gap_ext


#include <cfloat>

#include <iostream>
using std::cout;
using std::endl;


#include "semiglobal.h"
#include "alignment.h"
#include "matrix.h"


// Allocates space for the tables used in the semi-global alignment
// for efficiency reasons, the space for these tables is not released,
// although they will grow in size as needed and will occupy the max 
// allocated space throughout the use of the algorithm (however they 
// will only expose the section needed by the current alignment)
//
SemiGlobalAlign::SemiGlobalAlign() 
{
}


// resizes the tables used by the algorithm based on 
// the lengths of the "sequences" being aligned
void SemiGlobalAlign::resize(int rows, int cols)
{
  for (int i = 0; i < 3; ++i) {
    D[i].resize(rows, cols);
    M[i].resize(rows, cols);
  }
}


// the standard intialization with no initial gap penalties
void SemiGlobalAlign::init()
{
  D[_S_](0, 0) = -1;
  D[_H_](0, 0) = -1;
  D[_V_](0, 0) = -1;

  M[_S_](0, 0) = 0;
  M[_H_](0, 0) = 0;
  M[_V_](0, 0) = 0;
  
  int m = M[_S_].rows();
  int n = M[_S_].cols();
  
  for (int i = 1; i < m; i++) {
    D[_S_](i, 0) = V_to_V;
    D[_H_](i, 0) = V_to_V;
    D[_V_](i, 0) = V_to_V;

    M[_S_](i, 0) = DBL_MIN;
    M[_H_](i, 0) = DBL_MIN;
    M[_V_](i, 0) = 0;//-(Params::GAP_OPEN + (i-1)*Params::GAP_EXT);
  }
  
  for (int j = 1; j < n; j++) {
    D[_S_](0, j) = H_to_H;
    D[_H_](0, j) = H_to_H;
    D[_V_](0, j) = H_to_H;

    M[_S_](0, j) = DBL_MIN;
    M[_H_](0, j) = 0;//-(Params::GAP_OPEN + (j-1)*Params::GAP_EXT);
    M[_V_](0, j) = DBL_MIN;
  }
}

  

