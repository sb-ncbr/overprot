

#ifndef __ALIGNMENT_H__
#define __ALIGNMENT_H__


// Representation of pairwise alignment information. Contains two arrays that
// hold the indices of the aligned Ca atoms (or GAP_INDEX) to denote a gap.


class Protein;



const int GAP_INDEX = -1;


struct Alignment
{
public:
  int    capacity;       // max size reserved for the alignment (may not use all)
  int    size;           // the number of cells used to describe the alignment
  int*   first;          // the indices of Ca atoms in the first protein (or GAP_INDEX)
  int*   second;         // the indices of Ca atoms in the second protein (or GAP_INDEX)
  double  score;         // the alignment score (e.g. computed by dynamic programming)
  int    matches;        // the number of matched cells
  

  // creates an empty alignment of the given cpapcity
  Alignment(int n);

  // release the alignment resources
  ~Alignment();

  // resizes the alignment to given capacity; ignored if new capacity is smaller
  void resize(int n);

  // computes the rmsd described by the this alignment for the given proteins
  double computeRMSD(Protein* prot1, Protein* prot2) const;

  
private:
  // allocates enough space for the alignment
  void allocate(int n);

  // releases the alignment resources
  void release();
};


#endif

