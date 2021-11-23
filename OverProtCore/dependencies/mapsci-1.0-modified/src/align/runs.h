

#ifndef __RUNS_H__
#define __RUNS_H__


// Representation of *runs*, i.e. segments of matched atoms between two proteins
//
// class Run is a simple struct that stores the details of an individual run:
// * size (of the run (# of matched Ca atoms)
// * index within the alignment where a run starts and ends for *first* protein
// * index within the alignment where a run starts and ends for *second* protein
// * rotation and translation that best superpose
//   the matched Ca atoms (prot1 onot prot2)
//
// class RunsList represents a list of runs. A RunsList can grow in capacity
// (does not shrink) -- a RunsList gets reused throughout the iterations of
// the pairwise structure alignment algorithm.
//
// The most significant method is buildContiguous which takes an alignment
// and the two proteins and identifies the segments (*runs*) of matched Ca
// atoms and their optimal transformations.


#include "point.h"
#include "matrix.h"

class Protein;
struct Alignment;



struct Run
{
  int size;
  int start1, end1;
  int start2, end2;
  
  Matrix<double>  rot;
  Point           trans;
  
  Run();
};



class RunsList
{
public:
  // creates an empty list with the given intial capacity
  RunsList(int n);
  
  ~RunsList();

  // returns the size of the list
  int size() const;

  // read access to the i-th run in the list
  const Run& operator[](int i) const;

  // given a paiwise alignment identifies the segments (runs)
  // of matched CA atoms and populates the list
  void buildContiguous(Alignment* align,
		       Protein* prot1,
		       Protein* prot2,
		       int extend
		       );
  
private:
  // grows the list, if needed (does not shrink); 
  // the list is cleared implicitly (i.e. size is set to 0)
  void resize(int n);

  // allocates the requested amount
  void allocate(int n);

  // releases the list resources
  void release();

  
private:
  int   _capacity;
  int   _size;
  
  Run*  runs;
};


#endif

