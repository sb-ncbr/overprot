

// Encapsulates the computation of SVD, which currently is done via the
// ALGLIB project of Sergey Bochkanov, Vladimir Bystritsky.
//
// The ALGLIB website is:  http://www.alglib.net/
//
// The SVD source is at:

//   http://www.alglib.net/translator/dl/matrixops.general.svd.cpp.zip


#include <assert.h>

#include <iostream>
#include <vector>
using std::vector;


#include "ap.h"
#include "svd.h"

#include "msvd.h"
#include "point.h"
#include "matrix.h"



// Computes the transfomations (rotation and translation) that
// would suprpose the points in set A on set B. Point p in A
// is transformed to point q in B via:
//
//             q = rot * p + trans
//
// The rotaion and translation are returned through
// the last two reference parameters.
//
void MSVD::svd(const vector<Point>& A,    // input point sets A and B
	       const vector<Point>& B,    
	       int s1,                    // start and end index in A, inclusive
	       int e1,                    
	       int s2,                    // start and end index in B, inclusive
	       int e2,                   
	       Matrix<double>& R,         // the overall transformations
	       Point& T)
{
  assert(e1-s1+1 == e2-s2+1);   // are the ranges of the same size
    
  int n = e1-s1+1;

  // compute the means of the two point sets

  Point mean1 = Point(0, 0, 0);
  Point mean2 = Point(0, 0, 0);
  for (int k = 0; k < n; k++) {
    mean1 += A[s1+k];
    mean2 += B[s2+k];
  }
  mean1 /= n;
  mean2 /= n;
    

  // compute the covariance matrix of the translated point sets

  ap::real_2d_array ap_cvm;
  ap_cvm.setlength(3, 3);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double sum = 0;
      for (int k = 0; k < n ; k++) {
	sum +=  (B[s2+k][i] - mean2[i]) * (A[s1+k][j] - mean1[j]);
      }
      ap_cvm(i, j) = sum;
    }
  }


  // compute the SVD of the covariance matrix
    
  ap::real_1d_array  ap_q;
  ap_q.setlength(3);
    
  ap::real_2d_array ap_U;
  ap_U.setlength(3, 3);
    
  ap::real_2d_array ap_Vt;
  ap_Vt.setlength(3, 3);
    
  bool result = rmatrixsvd(ap_cvm, 3, 3, 2, 2, 2, ap_q, ap_U, ap_Vt);
  if (!result) {
    std::cout << "unable to compute svd" << std::endl;
      
    // return identity transformations
    R = Matrix<double>::identity(3);
    T = Point(0, 0, 0);
    return;
  }


  // compute the rotation matrix -- convert to our Matrix representation,
  // which has overloaded * operator and determinant computation
    
  Matrix<double> U = from_2d_array(ap_U);
  Matrix<double> Vt = from_2d_array(ap_Vt);

  assert(U.rows() == 3 && U.cols() == 3);
  assert(Vt.rows() == 3 && Vt.cols() == 3);
    
  Matrix<double> UVt = U*Vt;
  Matrix<double> D(3, 3);
  D(0, 0) = 1; D(1, 1) = 1; D(2, 2) = UVt.det3x3();
    
  R = U*D*Vt;
  T = mean2 - R*mean1;
}


// converts an ALGLIB 2D array to our Matrix representation
Matrix<double> MSVD::from_2d_array(const ap::real_2d_array& ap_M)
{
  int rows = ap_M.gethighbound(1) + 1;    // ap arrays return index of
  int cols = ap_M.gethighbound(2) + 1;    // last element, not size

  Matrix<double> M(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      M(i, j) = ap_M(i, j);
    }
  }

  return M;
}
