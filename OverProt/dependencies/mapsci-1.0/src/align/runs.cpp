

#include <assert.h>


#include "runs.h"
#include "msvd.h"
#include "alignment.h"
#include "protein.h"



Run::Run() : size(0),
	     start1(-1), end1(-1),
	     start2(-1), end2(-1),
	     rot(), trans()
{
}



// Given an alignment of prot1 and prot2 extracts segments of
// contiguous matches (each segment is called a run)
void RunsList::buildContiguous(Alignment* align,
			       Protein* prot1,
			       Protein* prot2,
			       int extend
			       )
{
  // the number of contiguous runs cannot exceed size of smaller protein
  int size = std::min(prot1->size(), prot2->size());
  this->resize(size);
  
  int count = 0;     // how many runs found so far
  int i = 0;         // index of current pair within the alignment
  
  while(i < align->size ) {
    // skip through pairs where one component is a gap
    while( (i < align->size) && ( (align->first[i] == GAP_INDEX) ||
				  (align->second[i] == GAP_INDEX)  ) ) {
      i++;
    }
    int start = i; // starting point of current run

    if (start == align->size) {
      break;
    }

    // skip through the matched pairs of current run
    while( (i < align->size) && ( (align->first[i] != GAP_INDEX) &&
				  (align->second[i] != GAP_INDEX)  ) ) {
      i++;
    }
    int end = i-1; // ending point of current run
 
    int cur_size = end - start + 1 + extend;
      
    if (cur_size >= 3) {
      assert(align->first[start] < prot1->size());
      assert(align->second[start] < prot2->size());
      assert(align->first[end] + extend < prot1->size());
      assert(align->second[end] + extend < prot2->size());
	
      // fill in run details: start,end index in each protein, and run length
      runs[count].start1 = align->first[start];
      runs[count].start2 = align->second[start];
	
      runs[count].end1 = align->first[end] + extend;
      runs[count].end2 = align->second[end] + extend;
      
      runs[count].size = cur_size;
	
      // compute rotation matrix that superposes the atoms of prot1 onto the
      // matches in prot2 (the optimal transformations are implicitly stored
      // in the run via the last parameters of SVD)
      //
      MSVD::svd(prot1->atoms,
		prot2->atoms,
		runs[count].start1, runs[count].end1,
		runs[count].start2, runs[count].end2,
		runs[count].rot,
		runs[count].trans);
      
      count++;
    }
  }
    
  // total number of runs found
  this->_size = count;
}


// creates an empty list with the given intial capacity
RunsList::RunsList(int n)
{
  allocate(n);
}


RunsList::~RunsList()
{
  release();
}


// returns the size if the list
int RunsList::size() const
{
  return _size;
}


// read access to the i-th run in the list
const Run& RunsList::operator[](int i) const
{
  return runs[i];
}


// grows the list, if needed (does not shrink); 
// the list is cleared implicitly (i.e. size is set to 0)
void RunsList::resize(int n)
{
  if (n > _capacity) {
    release();
    allocate(n);
  }
}


// allocates the requested amount
void RunsList::allocate(int n)
{
  runs = new Run[n];
  
  _capacity = n;
  _size = 0;
}


// releases the list resources
void RunsList::release()
{
  delete [] runs;
}
