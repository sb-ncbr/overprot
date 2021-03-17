

#ifndef __PARAMS_H__
#define __PARAMS_H__


// Parameters for fine-tuning the structure alignment algotrithm.


#include <string>
using std::string;



class Params
{
public:
  
  // parameters for the **pairwise alignment** algorithm
  // described in Ye et al., JBCB, 2004

  static double GAP_PENALTY;    // for alignment based on coordinates

  static double GAP_OPEN;       // for alignment based on angle triplets
  static double GAP_EXT;
  static double K;              // for angle triplet distance adjustment
  
  static double RUNS_ROT;       // for frobenius norm similarity of rot matrices
                                // of consistent run segments (see CliqueFinder)


  // parameters for the **multiple alignment** algorithm
  // described in Ye et al., WABI, 2006

  static double RHO;            // threshold for convergence rate
  static int MAX_ITERS;         // max number of iterations allowed


  // load the parameters from file which contains
  // one line per parameter value, e.g.:
  //    GAP_OPEN 6.0
  //    GAP_EXT 0.1
  static void load(const string& filename);

  // displays the current parameter values
  static void print();
};


#endif
