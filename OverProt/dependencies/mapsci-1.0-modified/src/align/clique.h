
#ifndef __CLIQUE_FINDER_H__
#define __CLIQUE_FINDER_H__


// Class for finding a clique of consistent runs of max weight (max number
// of atoms). Consistent runs are those that can be transformed/aligned with
// the ~same~ matrix.


#include <vector>
using std::vector;


#include "matrix.h"
class RunsList;
class Protein;



class CliqueFinder
{
public:

  CliqueFinder();

  
  // construct the connectivity graph of the runs; two runs are connected
  // if the frobenius norm of their rotation matrices is "small"
  //
  void buildGraph(const RunsList& runs, Protein* prot1, Protein* prot2);


  // finds the node with the largest weight
  int findMaxNode(const RunsList& runs);

  
  // checks whether the graph is empty
  bool isEmpty();


  // finds the clique of consistent runs of max weight
  // (i.e. runs whose rotation martiecs are similar)
  //
  vector<int> consistent(const RunsList& runs, Protein *prot1, Protein* prot2);


private:
  // these are defined as data members to avoid creating/destroying the graphs
  // during the iterations; normally, this class will be instantiated only once

  Matrix<short> graph;        // represents consistent runs
  Matrix<short> tempGraph;    // for temporary storage
};
  

#endif
