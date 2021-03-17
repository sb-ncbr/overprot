

#ifndef __SEMIGLOBAL_ALIGN_H__
#define __SEMIGLOBAL_ALIGN_H__


// Implementation of the dynamic programming algorithm for semiglobal alignment.
// Gaps are assigned an affine penalty g(n) = gap_open + n*gap_ext


#include <algorithm>


#include "matrix.h"
#include "params.h"
#include "alignment.h"


// path directions
const int _H_  = 0;
const int _S_  = 1;
const int _V_  = 2;

const int S_to_S  = 3;
const int S_to_H  = 4;
const int S_to_V  = 5;

const int H_to_H  = 6;
const int H_to_S  = 7;
const int H_to_V  = 8;

const int V_to_V  = 9;
const int V_to_S  = 10;
const int V_to_H  = 11;


class SemiGlobalAlign
{
private:


public:
  // Allocates space for the tables used in the semi-global alignment.
  // For efficiency reasons, the space for these tables is not released,
  // although they will grow in size as needed and will occupy the max 
  // allocated space throughout the use of the algorithm (however they 
  // will only expose the section needed by the current alignment)
  //
  SemiGlobalAlign();


  // the main method for performing the alignment:
  // takes two sequences of items and a distance function that can
  // compute the distance (as *double*) between two individual items
  //
  // a pointer to a previously allocated alignment must be provided as the
  // last parameter -- this will be modified to contain the final alignment
  //
  template <class SeqType, class DistFunc>
  void align(const SeqType& firstSeq,
	     const SeqType& secondSeq,
	     DistFunc distFunc,           // double (*distFunc)(ElmntType, ElmntType)
	     Alignment* ali);


private:
  // resizes the tables used by the algorithm based on 
  // the lengths of the "sequences" being aligned
  void resize(int rows, int cols);


  // the standard intialization with no initial gap penalties
  void init();


  // the table-filling section: takes two sequences of items and a distance
  // function that can compute the distance (as *double*) between two items
  //
  template <class SeqType, class DistFunc>
  void fill(const SeqType& firstSeq,
	    const SeqType& secondSeq,
	    DistFunc distFunc            // double (*distFunc)(ElmntType, ElmntType)
	    );

  
  template <class SeqType, class DistFunc>
  void build(Alignment* align,
	     const SeqType& firstSeq,
	     const SeqType& secondSeq,
	     DistFunc distFunc            // double (*distFunc)(ElmntType, ElmntType)
	     );
  
  
private:
  Matrix<int> D[3];      // the path (directions) matrix
  Matrix<double> M[3];   // the current scores (values) matrix
};



// the main method for performing the alignment:
// takes two sequences of items and a distance function that can
// compute the distance (as *double*) between two individual items
//
// a pointer to a previously allocated alignment must be provided as the
// last parameter -- this will be modified to contain the final alignment
//
template <class SeqType, class DistFunc>
void SemiGlobalAlign::align(const SeqType& firstSeq,
			    const SeqType& secondSeq,
			    DistFunc distFunc,           // double (*distFunc)(ElmntType, ElmntType)
			    Alignment* ali)
{
  int m = firstSeq.size() + 1;   // +1 to account for the extra row,col in
  int n = secondSeq.size() + 1;  // the DP matrices corresponding to gaps

  resize(m, n);
  init();
  fill(firstSeq, secondSeq, distFunc);
  build(ali, firstSeq, secondSeq, distFunc);
}


// the table-filling section: takes two sequences of items and a distance
// function that can compute the distance (as *double*) between two items
//
template <class SeqType, class DistFunc>
void SemiGlobalAlign::fill(const SeqType& firstSeq,
			   const SeqType& secondSeq,
			   DistFunc distFunc            // double (*distFunc)(ElmntType, ElmntType)
			   )
{
  int m = M[_S_].rows();
  int n = M[_S_].cols();
  
  for (int i = 1; i < m; i++) {
    for (int j = 1; j < n; j++) {

      double gap_open = (j == n-1) ? 0 : Params::GAP_OPEN;
      double gap_ext  = (j == n-1) ? 0 : Params::GAP_EXT;

      double v1 = M[_V_](i-1, j) - gap_ext;
      double v2 = M[_S_](i-1, j) - gap_open;
      double v3 = M[_H_](i-1, j) - gap_open;

      M[_V_](i, j) = std::max(v1, std::max(v2, v3));
      if (M[_V_](i, j) == v1) {
	D[_V_](i, j) = V_to_V;
      }
      else if (M[_V_](i, j) == v2) {
	D[_V_](i, j) = V_to_S;
      }
      else {
	D[_V_](i, j) = V_to_H;
      }


      gap_open = (i == m-1) ? 0 : Params::GAP_OPEN;
      gap_ext  = (i == m-1) ? 0 : Params::GAP_EXT;
      
      v1 = M[_H_](i, j-1) - gap_ext;
      v2 = M[_S_](i, j-1) - gap_open;
      v3 = M[_V_](i, j-1) - gap_open;

      M[_H_](i, j) = std::max(v1, std::max(v2, v3));
      if (M[_H_](i, j) == v1) {
	D[_H_](i, j) = H_to_H;
      }
      else if (M[_H_](i, j) == v2) {
	D[_H_](i, j) = H_to_S;
      }
      else {
	D[_H_](i, j) = H_to_V;
      }


      double dist = Params::K - distFunc(firstSeq[i-1], secondSeq[j-1]);

      v1 = M[_V_](i-1, j-1) + dist;
      v2 = M[_H_](i-1, j-1) + dist;
      v3 = M[_S_](i-1, j-1) + dist;
      M[_S_](i, j) = std::max(v1, std::max(v2, v3));

      if (M[_S_](i, j) == v3) {
	D[_S_](i, j) = S_to_S;
      }
      else if (M[_S_](i, j) == v2) {
	D[_S_](i, j) = S_to_H;
      }
      else {
	D[_S_](i, j) = S_to_V;
      }
    }
  }
}



// traverses the direction matrix to recreate the alignment
template <class SeqType, class DistFunc>
void SemiGlobalAlign::build(Alignment* align,
			    const SeqType& firstSeq,
			    const SeqType& secondSeq,
			    DistFunc distFunc            // double (*distFunc)(ElmntType, ElmntType)
			    )
{
  int m = D[_S_].rows();
  int n = D[_S_].cols();
  
  // longest possible size for alignment is rows+cols-1
  align->resize(m+n);
  
  int i = m-1;
  int j = n-1;
  int k = -1;
  int count = 0;
  
  if (M[_S_](i, j) >= M[_H_](i, j) && M[_S_](i, j) >= M[_V_](i, j)) {
    k = _S_;
  }
  else if (M[_V_](i, j) >= M[_S_](i, j) && M[_V_](i, j) >= M[_H_](i, j)) {
    k = _V_;
  }
  else {
    k = _H_;
  }
  
  int matches = 0;
  
  while ( (i > 0) || (j > 0) ) {
    
    if (D[k](i, j) == S_to_S) {
      assert(k == _S_);
      i--;
      j--;
      align->first[count]  = i;
      align->second[count] = j;
      
      ++matches;
      
/*       double dist = distFunc(firstSeq[i], secondSeq[j]); */
/*       cout << "dist: " << dist << ", " << firstSeq[i] << ", " << secondSeq[j] << endl; */
    }
    else if (D[k](i, j) == S_to_H) {
      assert(k == _S_);
      k = _H_;
      j--;
      align->first[count]  = GAP_INDEX;
      align->second[count] = j;
    }
    else if (D[k](i, j) == S_to_V) {
      assert(k == _S_);
      k = _V_;
      i--;
      align->first[count]  = i;
      align->second[count] = GAP_INDEX;
    }
    
    else if (D[k](i, j) == H_to_H) {
      //      assert(k == _H_);
      j--;
      align->first[count]  = GAP_INDEX;
      align->second[count] = j;
    }
    else if (D[k](i, j) == H_to_S) {
      assert(k == _H_);
      k = _S_;
      j--;
      align->first[count]  = GAP_INDEX;
      align->second[count] = j;
    }
    else if (D[k](i, j) == H_to_V) {
      assert(k == _H_);
      k = _V_;
      i--;
      align->first[count]  = i;
      align->second[count] = GAP_INDEX;
    }
    
    else if (D[k](i, j) == V_to_V) {
      //      assert(k == _V_);
      i--;
      align->first[count]  = i;
      align->second[count] = GAP_INDEX;
    }
    else if (D[k](i, j) == V_to_S) {
      assert(k == _V_);
      k = _S_;
      i--;
      align->first[count]  = i;
      align->second[count] = GAP_INDEX;
    }
    else if (D[k](i, j) == V_to_H) {
      assert(k == _V_);
      k = _H_;
      j--;
      align->first[count]  = GAP_INDEX;
      align->second[count] = j;
    }
    
    else {
      cout << "semiglobal: invalid direction D[" << k << "](" << i << ", " << j << ") = " 
	   << D[k](i, j) << endl;
    }

    count++;
  }

/*   cout << "matches, m, n: " << matches << ", " << m << ", " << n << endl; */

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
  align->score = M[_S_](m-1, n-1);
  align->matches = matches;
}


#endif
