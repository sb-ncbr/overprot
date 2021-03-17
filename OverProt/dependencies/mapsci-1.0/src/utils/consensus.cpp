

#include <cfloat>
#include <climits>

#include <iostream>
#include <vector>
#include <string>
using std::cout;
using std::cin;
using std::cerr;
using std::endl;
using std::vector;
using std::string;


#include "consensus.h"
#include "alignment.h"
#include "protein.h"
#include "pairwise.h"
#include "multiple.h"
#include "zscore.h"
#include "summary.h"



// selects the protein with overall smallest sum of pairwise alignment scores
Protein* Consensus::center(const vector<Protein*>& M)
{
  int indJ = -1;
  double center_sum = DBL_MAX;
  
  PairAlign pair;
  Alignment* align = new Alignment(100);
  
  for (int i = 0; i < M.size(); ++i) {
    double sum = 0;
    for (int j = 0; j < M.size(); ++j) {
      if (i == j) {
	continue;
      }
      pair.align(M[j], M[i], align);
      sum += align->score;
    }
    if (sum < center_sum) {
      center_sum = sum;
      indJ = i;
    }
  }
  delete align;

  cout << "consensus: center, " << indJ << endl << endl;

  Protein* J = M[indJ]->clone();

  return J;
}


// selects the protein with overall smallest maximum pairwise alignment scores
Protein* Consensus::minmax(const vector<Protein*>& M)
{
  int indJ = -1;
  double center_minmax = DBL_MAX;

  PairAlign pair;
  Alignment* align = new Alignment(100);
  
  for (int i = 0; i < M.size(); ++i) {
    double cur_max = -DBL_MAX;
    for (int j = 0; j < M.size(); ++j) {
      if (i == j) {
	continue;
      }
      pair.align(M[i], M[j], align);
      if (align->score > cur_max) {
	cur_max = align->score;
      }
    }
    if (cur_max < center_minmax) {
      center_minmax = cur_max;
      indJ = i;
    }
  }
  delete align;

  cout << "consensus: minmax, " << indJ << endl << endl;
  
  Protein* J = M[indJ]->clone();
  
  return J;
}


// selects the protein that gives the largest initial core
Protein* Consensus::maxcore(const vector<Protein*>& M)
{
  int max_core = INT_MIN;
  int indJ = -1;

  PairAlign pair;
  MultiAlign multi;

  vector<Alignment*> MA(M.size());
  for (int i = 0; i < M.size(); ++i) {
    MA[i] = new Alignment(100);
  }
  
  for (int i = 0; i < M.size(); ++i) {
    for (int j = 0; j < MA.size(); ++j) {
      pair.align(M[j], M[i], MA[j]);
    }
    multi.center_star(MA, M[i]);
    
    double rmsd = -1;
    int core = -1;
    compute_core(MA, M, Params::GAP_PENALTY, rmsd, core);

    if (core > max_core) {
      max_core = core;
      indJ = i;
    }
  }

  int min_size, max_size, ave_size;
  length_stats(M, min_size, max_size, ave_size);
  
  double core_pct = double(max_core) / min_size;
  if (core_pct < .15) {
    return Consensus::median(M);
  }
  
  cout << "consensus: maxcore, " << indJ << endl << endl;

  Protein* J = M[indJ]->clone();

  return J;
}


// selects the protein of median length
Protein* Consensus::median(const vector<Protein*>& M)
{
  vector<int> sizes(M.size());
  for (int i = 0; i < sizes.size(); ++i) {
    sizes[i] = M[i]->size();
  }
  
  sort(sizes.begin(), sizes.end());
  
  int indJ = -1;
  int center_size = sizes[ M.size() / 2 ];
  for (int i = 0; i < M.size(); ++i) {
    if (M[i]->size() == center_size) {
      indJ = i;
      break;
    }
  }

  cout << "consensus: median, " << indJ << endl << endl;

  Protein* J = M[indJ]->clone();

  return J;
}


// selects the protein with hte given index
Protein* Consensus::index(const vector<Protein*>& M)
{
  int indJ = -1;
  
  cout << "index of consensus: ";
  cin >> indJ;

  cout << "consensus: median, " << indJ << endl << endl;

  Protein* J = M[indJ]->clone();

  return J;
}


// returns corresponding consensus given one of "ceneter", "minmax", "median"
Protein* Consensus::select(const vector<Protein*>& M,
			   const string& choice)
{
  if (choice == "center") {
    return Consensus::center(M);
  }
  else if (choice == "minmax") {
    return Consensus::minmax(M);
  }
  else if (choice == "maxcore") {
    return Consensus::maxcore(M);
  }
  else if (choice == "median") {
    return Consensus::median(M);
  }
  else if (choice == "index") {
    return Consensus::index(M);
  }
  else {
    cerr << "consensus error: invalid consensus type" << choice << endl;
    exit(1);
  }
}
