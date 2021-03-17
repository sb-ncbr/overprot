

#ifndef __MSVD_H__
#define __MSVD_H__


// Encapsulates the computation of SVD, which currently is done via the
// ALGLIB project of Sergey Bochkanov, Vladimir Bystritsky.
//
// The ALGLIB website is:  http://www.alglib.net/
//
// The SVD source is at:  http://www.alglib.net/translator/dl/matrixops.general.svd.cpp.zip


#include <vector>
using std::vector;

#include "ap.h"


class Point;

template <class T>
class Matrix;



// M(apsci)SVD wrapper around the ALGLIB SVD
namespace MSVD
{
  // Computes the transfomations (rotation and translation) that
  // would suprpose the points in set A on set B. Point p in A
  // is transformed to point q in B via:
  //
  //             q = p * rot + trans
  //
  // The rotaion and translation are returned through
  // the last two reference parameters.
  //
  void svd(const vector<Point>& A,    // input point sets A and B
	   const vector<Point>& B,    
	   int s1,                    // the range of points in A (start index, end index, inclusive)
	   int e1,                    
	   int s2,                    // the range of points in B (start index, end index, inclusive)
	   int e2,                   
	   Matrix<double>& R,         // the transformations (returned by reference)
	   Point& T);

  
  
  // converts an ALGLIB 2D array to our Matrix representation
  Matrix<double> from_2d_array(const ap::real_2d_array& ap_M);
}


#endif
