

// Implementation of the algorithm for multiple structure alignment in:
//
// Ye at al., "Multiple Structure Alignment ...", WABI: 115-125 (2006)


#include <assert.h>
#include <cfloat>
#include <cstdio>

#include <vector>
#include <utility>
using std::vector;
using std::pair;


#include "multiple.h"
#include "params.h"
#include "alignment.h"
#include "pairwise.h"
#include "global.h"
#include "consensus.h"

#include "protein.h"
#include "matrix.h"
#include "point.h"
#include "msvd.h"




// By default space for 100x100 is reserved for the center-star alignment, but
// it is resized as needed for each new alignment; for efficiency reasons, the
// space is not released and starALign will occupy the max allocated space throughout
// the use of the algorithm (however it will only expose the section currently needed)
//
MultiAlign::MultiAlign() : starAlign(100, 100)
{
}


// the main method for performing the multiple alignment; the alignment and
// the consensus protein are returned as a std::pair; input is the protein set 
// the type of initial consensus to try (see consensus.c)
//
MultiAlign::RT  MultiAlign::align(vector<Protein*>& M, 
				  const string& cons_type)
{
  PairAlign pair;
  GlobalAlign global;
  

  // Step 1: consensus selction
  Protein* J = Consensus::select(M, cons_type);
  J->filename = "consensus.pdb";
  
  // Step 2: pairwise structure align each protein with consensus
  vector<Alignment*> MA(M.size());
  for (int i = 0; i < M.size(); ++i) {
    MA[i] = new Alignment(100);
    pair.align(M[i], J, MA[i]);
  }

  
  int count = 0;
  double cur_score = DBL_MAX;
  double ratio = DBL_MAX;

  do
  {
    // Step 6, 7, 8, 9
    center_star(MA, J);
    transform(M, MA, J);
    double new_score = multi_score(M, MA, J);
    compute_consensus(M, MA, J);

    // convergence test
    ratio = fabs((new_score - cur_score) / cur_score);
    cur_score = new_score;
    ++count;

    if (ratio > Params::RHO) {
      for (int i = 0; i < M.size(); ++i) {
	global.align(M[i]->atoms, J->atoms, Point::squaredDistance, MA[i]);
      }
    }

    printf("Iteration %2d:\t%.6f\n", count, cur_score);
  }  
  while( (ratio > Params::RHO) && (count < Params::MAX_ITERS) );
    
  RT result(MA, J);
  
  return result;
}


// given a list of alignments (MA) inserts gaps in each alignment in a center-star
// fashion so that they are all of the same length and maintain their original
// correspondence with the consensus protein (for each alignment MA[i], MA[i]->second
// contains the indices of atoms from the current consensus)
//
void MultiAlign::center_star(vector<Alignment*>& MA,
			     Protein* J)
{
  //num_gap_regions -- represents the # of gap intervals in the consensus protein,
  //                   i.e. each Ca atom is preceded by a region of 0 or more gaps;
  //                   there is an extra gap region that corresponds to the number
  //                   of gaps needed to pad at the end -- past last Ca atom)
  //
  //starAlign[i][j] -- represents the number gaps before residue j in protein i alignment

  int num_gap_regions = J->size() + 1;  //+ 1 to account for gap region after last Ca atom

  starAlign.resize(MA.size(), num_gap_regions);
  

  //max_gaps[i] -- the max length of the gap regions before Ca atom i over all alignments
  //
  vector<int> max_gaps(num_gap_regions);


  for (int i = 0; i < MA.size(); ++i) {
  
    // find the number of gaps preceding each Ca atom in the consensus for the current alignment;
    // cur_ind and prev_ind mark a region of gaps in consensus and num_gaps = (cur_ind - prev_ind - 1)
    //
    int pos = 0;
    int prev_ind = -1;
    for (int j = 0; j < MA[i]->size; ++j) {
      if (MA[i]->second[j] != GAP_INDEX) {
	starAlign(i, pos) = j - prev_ind - 1;
	prev_ind = j;
	pos++;
      }
    }
    starAlign(i, pos++) = MA[i]->size - prev_ind - 1;   // padding after last Ca atom
    assert(pos == num_gap_regions);


    //update the max length of all gap regions
    for (int j = 0; j < num_gap_regions; ++j) {
      max_gaps[j] = (max_gaps[j] > starAlign(i, j)) ? max_gaps[j] : starAlign(i, j);
    }
  }


  // find out the new total length for the multiple alignment correspondence
  // after all pairwise alignments have been padded to have equal lengths
  int new_size = J->size();
  for (int j = 0; j < num_gap_regions; ++j) {
    new_size += max_gaps[j];
  }

  assert(new_size > 0);
  
  // this loop inserts extra gaps in each protein alignment, so that all have the same length
  for (int i = 0; i < MA.size(); ++i) {

    // this will become the new alignment by copying from the original and adding gaps
    Alignment* temp = new Alignment(new_size);
    temp->size = new_size;
    temp->score = MA[i]->score;                // same score since only pair of gaps inserted
    
    int align_pos = 0;   // the index of the cell being copied from original alignment
    int temp_pos = 0;    // the index where the cell is copied in the new (temp) array
    for (int j = 0; j < num_gap_regions-1; ++j) {
      // add extra gaps
      int extra_gaps = max_gaps[j] - starAlign(i, j);
      for (int k = 0; k < extra_gaps; ++k) {
	temp->first[temp_pos] = GAP_INDEX;
	temp->second[temp_pos] = GAP_INDEX;
	temp_pos++;
      }
      // transfer the gaps from the orginal alignment moving in tandem in both alignments
      for (int k = 0; k < starAlign(i, j); ++k) {
	temp->first[temp_pos] = MA[i]->first[align_pos];
	temp->second[temp_pos] = MA[i]->second[align_pos];
	temp_pos++;
	align_pos++;
      }
      // copy wahtever is matched with current Ca atom in consensus (could be a gap)
      temp->first[temp_pos] = MA[i]->first[align_pos];
      temp->second[temp_pos] = MA[i]->second[align_pos];
      temp_pos++;
      align_pos++;
    }

    // transfer any left over cells up to the length of the alignment
    while(align_pos < MA[i]->size) {
      temp->first[temp_pos] = MA[i]->first[align_pos];
      temp->second[temp_pos] = MA[i]->second[align_pos];
      temp_pos++;
      align_pos++;
    }
    // pad with gaps at the end of the alignment to ensure equal size
    while(temp_pos < new_size) {
      temp->first[temp_pos] = GAP_INDEX;
      temp->second[temp_pos] = GAP_INDEX;
      temp_pos++;
    }

    // replace the old alignment with the extended one
    delete MA[i];
    MA[i] = temp;
  }
}



// transforms each protein in the data set so that it best lines up
// with the consensus; the transformation matrix is determined based
// on the Ca atoms that are matched by the current alignment
//
void MultiAlign::transform(const vector<Protein*>& M,
			   const vector<Alignment*>& MA,
			   Protein* J)
{
  // temporary vectors to collect matched Ca atoms in current protein and consensus
  // (overall cannot exceed the size of the consensus)
  vector<Point> protAtoms(J->size());
  vector<Point> consAtoms(J->size());
  

  // find the Rotation and Translation that align i-th protein and the consensus
  //
  for (int i = 0; i < M.size(); ++i) {

    // extract the matching atoms in the i-th alignment
    int n = 0;
    for (int j = 0; j < MA[i]->size; ++j) {
      int p = MA[i]->first[j];              // the index of the molecule in i-th protein
      int q = MA[i]->second[j];             // the index of the molecule in consensus protein
      if ( (p != GAP_INDEX) && (q != GAP_INDEX) ) {
	protAtoms[n] = M[i]->atoms[p];
	consAtoms[n] = J->atoms[q];
	
	n++;
      }
    }

    if (n < 3) {
      continue;
    }
    
    Matrix<double> rot;
    Point trans;
    MSVD::svd(protAtoms, consAtoms, 0, n-1, 0, n-1, rot, trans);
    
    M[i]->orient(rot, trans);
  }
}


// computes the new consensus based on the transformed coordinates and current alignment
// for more details see Theorem 1, in Ye et al. WABI paper
//
void MultiAlign::compute_consensus(const vector<Protein*>& M,
				   const vector<Alignment*>&  MA,
				   Protein* J)
{
  //at this point all alignments should have the same size
  int columns = MA[0]->size;
  J->atoms = vector<Point>();
    
  for (int j = 0; j < columns; ++j) {   //for each column in the alignment
    
    int gaps = 0;        //the number of gap entries in column j
    int non_gaps = 0;    //the number of non-gap entries in column j

    //find the centroid of the non-gap entries in column j
    Point centroid(0, 0, 0); 
    for (int i = 0; i < M.size(); ++i) {    //for each protein
      int k = MA[i]->first[j];     //the index of the molecule in i-th protein
      if (k != GAP_INDEX) {
	centroid += M[i]->atoms[k];
	non_gaps += 1;
      }
    }
    
    //skip columns consisting only of gaps
    if (non_gaps == 0) {
      continue;
    }
    
    centroid /= non_gaps;

    gaps = M.size() - non_gaps;
    
    //find the sum of the square distances of the non-gap atoms to the centroid
    //
    double sum = 0;
    for (int i = 0; i < M.size(); ++i) {    //for each protein
      int k = MA[i]->first[j];     //the index of the molecule in i-th protein
      if (k != GAP_INDEX) {
	sum += Point::squaredDistance(centroid, M[i]->atoms[k]);
      }
    }
    
    // perform the test in Theorem 1 and add a Ca atom to consensus
    if ( sum < (Params::GAP_PENALTY*(non_gaps - gaps)) ) {
      J->atoms.push_back(centroid);
    }
  }
}


// computes the score of a pairwise alignment: p. 13, Ye et al., WABI paper
double MultiAlign::pair_score(Protein* prot1,
			      Protein* prot2,
			      Alignment* align)
{
  double score = 0;
  for (int j = 0; j < align->size; j++) {
    int index0 = align->first[j];
    int index1 = align->second[j];
    
    if ( (index0 != GAP_INDEX) && (index1 != GAP_INDEX) ) {
      Point atom1 = prot1->atoms[index0];
      Point atom2 = prot2->atoms[index1];
      double dist = Point::squaredDistance(atom1, atom2);
      
      score += dist;
    }
    else if ( (index0 == GAP_INDEX) && (index1 == GAP_INDEX) ) {
      continue;
    }
    else {
      score += Params::GAP_PENALTY;
    }
  }
  
  return score;
}


// computes the score of a multiple alignment: p. 14, Ye et al., WABI paper
// (essentially the sum of all pairwise alignment scores)
//
double MultiAlign::multi_score(const vector<Protein*>& M,
			       const vector<Alignment*>& MA,
			       Protein* J)
{
  assert(M.size() == MA.size());
  
  double score = 0;
  for (int i = 0; i < M.size(); ++i) {
    score += pair_score(M[i], J, MA[i]);
  }
  return score;
}


// convenience method for releasing a mutiple alignment result
void MultiAlign::free(const RT& ali)
{
  const vector<Alignment*>& MA = ali.first;
  for (int i = 0; i < MA.size(); ++i) {
    delete MA[i];
  }
  
  Protein* J = ali.second;
  delete J;
}


