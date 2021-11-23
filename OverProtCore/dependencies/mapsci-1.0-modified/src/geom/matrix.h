

#ifndef __MATRIX_H__
#define __MATRIX_H__


// A 2D matrix with some basic operations (matrix multiplication, point-matrix
// multiplication, frobenius norm).
//
// A matrix can be resized, but if the new dimensions are samller then the
// current ones, excess space is not released (this is for efficiency reasons
// since a matrix is used as the 2D table in the dynamic programming algorithms,
// so the table only grows as needed and space is re-used)


#include <assert.h>
#include <cmath>

#include <iostream>
#include <vector>
#include <algorithm>
using std::cout;
using std::endl;
using std::vector;


#include "point.h"


template <class T>
class Matrix
{
public:
  // creates a matrix of the given dimensions 3x3 by default
  Matrix(int newRows=3, int newCols=3) : matrix(),
					 _rows(newRows), _cols(newCols)
  {
    assert(_rows > 0);
    assert(_cols > 0);
    
    matrix = vector< vector<T> >(_rows);
    for (int r = 0; r < _rows; ++r) {
      matrix[r] = vector<T>(_cols);
    }
  }


  // creates a copy of the given (original) matrix
  Matrix(const Matrix<T>& orig) : matrix(),
				  _rows(orig.rows()), _cols(orig.cols())
  {
    assert(_rows > 0);
    assert(_cols > 0);
    
    matrix = vector< vector<T> >(_rows);
    for (int r = 0; r < _rows; ++r) {
      matrix[r] = vector<T>(_cols);
      std::copy(orig.matrix[r].begin(), orig.matrix[r].begin() + _cols,
		matrix[r].begin());
    }
  }


  // copies the given matrix, by resizing appropriately;
  // (excess space is not released)
  Matrix<T>& operator=(const Matrix<T>& orig)
  {
    assert(_rows > 0);
    assert(_cols > 0);

    assert(orig.rows() > 0);
    assert(orig.cols() > 0);
    
    if (this != &orig) {
      this->resize(orig.rows(), orig.cols());
      for (int r = 0; r < this->rows(); ++r) {
	std::copy(orig.matrix[r].begin(), orig.matrix[r].begin() + _cols,
		  matrix[r].begin());
      }
    }
    
    return *this;
  }


  // resizes the matrix to be of the given dimensions;
  // if the matrix is already big enough, space is not
  // released but the dimensions are adjusted
  //
  void resize(int newRows, int newCols)
  {
    assert(newRows > 0);
    assert(newCols > 0);

    if (_rows < newRows) {
      matrix.resize(newRows);
      for (int r = _rows; r < newRows; ++r) {
	matrix[r] = vector<T>(newCols);
      }
    }

    if (_cols < newCols) {
      for (int r = 0; r < _rows; ++r) {
	matrix[r].resize(newCols);
      }
    }

    _rows = newRows;
    _cols = newCols;
  }


  // number of rows for the current view
  int rows() const
  {
    return _rows;
  }

  // number of columns for the current view
  int cols() const
  {
    return _cols;
  }


  // read access to the matrix cell (i, j)
  const T& operator()(int i, int j) const
  {
    assert(i >= 0 && i < _rows);
    assert(j >= 0 && j < _cols);
    
    return matrix[i][j];
  }

  
  // write access to the matrix cell (i, j)
  T& operator()(int i, int j)
  {
    assert(i >= 0 && i < _rows);
    assert(j >= 0 && j < _cols);

    return matrix[i][j];
  }

  
  // computes the determinant of a given *3x3* matrix
  T det3x3()
  {
    assert( rows() == 3 );
    assert( cols() == 3 );

    const Matrix<T> &M = *this;
    T a = M(0, 0)*M(1, 1)*M(2, 2);
    T b = M(0, 1)*M(1, 2)*M(2, 0);
    T c = M(1, 0)*M(2, 1)*M(0, 2);

    T e = M(0, 2)*M(1, 1)*M(2, 0);
    T f = M(1, 0)*M(0, 1)*M(2, 2);
    T g = M(1, 2)*M(2, 1)*M(0, 0);
    
    return (a+b+c) - (e+f+g);
  }


  // prints a matrix repreesntion to standard output
  void print(const char* header = "") const
  {
    cout << header << endl;
    cout << "rows = " << rows() << "  cols = " << cols() << endl;
    const Matrix<T> &M = *this;
    for (int r = 0; r < rows(); ++r) {
      for (int c = 0; c < cols(); ++c) {
	cout << M(r, c) << '\t';
      }
      cout << endl;
    }
    cout << endl;
  }
  

  // returns a square identity matrix of the given dimensions 
  static Matrix<T> identity(int dims)
  {
    Matrix<T> M(dims, dims);
    for (int i = 0; i < dims; ++i) {
      M(i, i) = 1;
    }
    return M;
  }

  
private:
  vector< vector<T> > matrix;
  int _rows;
  int _cols;
};



// matrix multiplication
template <class T>
Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B)
{
  assert( A.cols() == B.rows() );
  
  Matrix<T> R(A.rows(), B.cols());
  
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < B.cols(); ++j) {
      T sum = 0;
      for (int k = 0; k < A.cols(); ++k) {
	sum += A(i, k)*B(k, j);
      }
      R(i, j) = sum;
    }
  }

  assert(R.rows() > 0);
  assert(R.cols() > 0);
  
  return R;
}


// left-multiplication of point by a matrix
template<class T>
Point operator*(const Matrix<T>& M, const Point& p)
{
  assert(M.rows() == 3);
  assert(M.cols() == 3);
  
  double coords[] = {p.x(), p.y(), p.z()};
  double newCoords[] = {0, 0, 0};
  
  for (int i = 0; i < 3; ++i) {
    double sum = 0;
    for (int j = 0; j < 3; j++) {
      sum += M(i, j)*coords[j];
    }
    newCoords[i] = sum;
  }
  
  return Point(newCoords[0], newCoords[1], newCoords[2]);
}


// Frobenius norm of the difference between matrices A and B
template <class T>
static double frobenius(const Matrix<T>& A, const Matrix<T>& B) 
{
  assert( A.rows() == B.rows() );
  assert( A.cols() == B.cols() );
  
  double norm = 0;
  for (int i = 0; i < A.rows(); i++) {
    for (int j = 0; j < A.cols(); j++) {
      double dx = (A(i, j) - B(i, j));
      norm += dx*dx;
    }
  }
  return sqrt(norm);
}


#endif
