

#ifndef __GLOBAL_ALIGN_H__
#define __GLOBAL_ALIGN_H__


// Implementation of the dynamic programming algorithm for global alignment.


#include "params.h"
#include "matrix.h"
struct Alignment;



class GlobalAlign
{
 private:
  // path directions
  static const int VERT  = 1;
  static const int HORI  = 2;
  static const int DIAG  = 3;
  
 public:
  // Allocates space for the tables used in the global alignment.
  // For efficiency reasons, the space for these tables is not released,
  // although they will grow in size as needed and will occupy the max 
  // allocated space throughout the use of the algorithm (however they 
  // will only expose the section needed by the current alignment)
  //
  GlobalAlign();


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
  

  // the standard intialization with initial gap penalties
  void init();


  // the table-filling section: takes two sequences of items and a distance
  // function that can compute the distance (as *double*) between two items
  //
  template <class SeqType, class DistFunc>
    void fill(const SeqType& firstSeq,
	      const SeqType& secondSeq,
	      DistFunc distFunc            // double (*distFunc)(ElmntType, ElmntType)
	      );


  // traversies the direction matrix to recreate the alignment
  void build(Alignment* align);


 private:
  Matrix<int> D;      // the path (directions) matrix
  Matrix<double> V;   // the current scores (values) matrix
};



// the main method for performing the alignment:
// takes two sequences of items and a distance function that can
// compute the distance (as *double*) between two individual items
//
// a pointer to a previously allocated alignment must be provided as the
// last parameter -- this will be modified to contain the final alignment
//
template <class SeqType, class DistFunc>
  void GlobalAlign::align(const SeqType& firstSeq,
			  const SeqType& secondSeq,
			  DistFunc distFunc,           // double (*distFunc)(ElmntType, ElmntType)
			  Alignment* ali)
{
  int m = firstSeq.size() + 1;   // +1 to account for the extra row,col in
  int n = secondSeq.size() + 1;  // the DP matrices corresponding to gaps

  resize(m, n);
  init();
  fill(firstSeq, secondSeq, distFunc);
  build(ali);
}


// the table-filling section: takes two sequences of items and a distance
// function that can compute the distance (as *double*) between two items
//
template <class SeqType, class DistFunc>
  void GlobalAlign::fill(const SeqType& firstSeq,
			 const SeqType& secondSeq,
			 DistFunc distFunc            // double (*distFunc)(ElmntType, ElmntType)
			 )
{
  for (int i = 1; i < V.rows(); i++) {
    for (int j = 1; j < V.cols(); j++) {
      double v1 = V(i, j-1) + Params::GAP_PENALTY;
      double v2 = V(i-1, j) + Params::GAP_PENALTY;

      double dist = distFunc(firstSeq[i-1], secondSeq[j-1]);
      //Point::squaredDistance(atom1, atom2);
	
      double v3 = V(i-1, j-1) + dist;
	
      if ( (v3 <= v1) && (v3 <= v2) ) {
	V(i, j) = v3;
	D(i, j) = DIAG;
      }
      else if ( (v1 <= v2) && (v1 <= v3) ) {
	V(i, j) = v1;
	D(i, j) = HORI;
      }
      else {
	V(i, j) = v2;
	D(i, j) = VERT;
      }
    }
  }
}
  

#endif
