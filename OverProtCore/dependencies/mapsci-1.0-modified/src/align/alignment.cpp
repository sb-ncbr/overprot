

#include <cmath>


#include "alignment.h"
#include "point.h"
#include "protein.h"
#include <stdio.h>
 #include <iostream>
using std::cout;


// creates an empty alignment of the given cpapcity
Alignment::Alignment(int n) : first(0), second(0)
{
  allocate(n);
  size = 0;
  score = -1;
}


// release the alignment resources
Alignment::~Alignment()
{
  release();
}


// resizes the alignment to given capacity; ignored if new capacity is smaller
void Alignment::resize(int n)
{
  if (n > capacity) {
    release();
    allocate(n);
  }
}


// computes the rmsd described by the this alignment for the given proteins
double Alignment::computeRMSD(Protein* prot1, Protein* prot2) const
{
  double rmsd = 0;
  int matches = 0;
    
  for (int i = 0; i < this->size; ++i) {
    int index0 = this->first[i];
    int index1 = this->second[i];
    if (index0 != GAP_INDEX && index1 != GAP_INDEX) {
      Point p = prot1->atoms[index0];
      Point q = prot2->atoms[index1];
      rmsd += Point::squaredDistance(p, q);
      matches++;
    }
  }

  // Modified by @Adam:
  rmsd = (matches == 0) ? -1 : sqrt(rmsd / matches);  
  // // Original wrong formula:
  // rmsd = (matches == 0) ? -1 : sqrt(rmsd) / matches;
    
  return rmsd;
}


// allocates enough space for the alignment
void Alignment::allocate(int n)
{
  first = new int[n];
  second = new int[n];

  capacity = n;
  size = 0;
  score = -1;
}


// releases the alignment resources
void Alignment::release()
{
  delete [] first;
  delete [] second;
}
