

#ifndef __MULTIPLE_H__
#define __MULTIPLE_H__


// Implementation of the algorithm for multiple structure alignment in:
//
// Ye at al., "Multiple Structure Alignment ...", WABI: 115-125 (2006)


#include <vector>
#include <string>
#include <utility>
using std::vector;
using std::string;
using std::pair;


#include "matrix.h"
struct Alignment;
class Protein;



class MultiAlign
{
public:
  typedef     pair< vector<Alignment*>, Protein* >    RT;
  
private:
  Matrix<int> starAlign;


public:
  MultiAlign();


  // performs multiple alignment for the given proteins and a choice
  // of initial consensus protein (see consensus.h)
  RT align(vector<Protein*>& M, 
	   const string& cons_type);


  // convenience method for releasing a mutiple alignment result
  static void free(const RT& MA);

  
public:
  // inserts gaps in each of the given alignments in a center-star
  // so that they are all of the same length and maintain their original
  // correspondence with the matched indices in the consensus protein 
  void center_star(vector<Alignment*>& MA,
		   Protein *J);


  // transforms each protein in the data set so that it best lines up
  // with the consensus; the transformation matrix is determined based
  // on the Ca atoms that are matched by the current alignment
  void transform(const vector<Protein*>& M,
		 const vector<Alignment*>& MA,
		 Protein* J);


  // computes the new consensus based on the transformed coordinates and the
  // current alignment as described in Theorem 1, in Ye et al. WABI paper
  void compute_consensus(const vector<Protein*>& M,
			 const vector<Alignment*>& MA,
			 Protein* J);
  

  // computes the score of a pairwise alignment: p. 13, Ye et al., WABI paper
  double pair_score(Protein* prot1,
		    Protein* prot2,
		    Alignment* align);
  

  // computes the score of a multiple alignment: p. 14, Ye et al., WABI paper
  // (essentially the sum of all pairwise alignment scores)
  double multi_score(const vector<Protein*>& M,
		     const vector<Alignment*>& MA,
		     Protein* J);
};


#endif
