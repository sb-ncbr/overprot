

#ifndef __CONSENSUS_H__
#define __CONSENSUS_H__


// Class for selecting an initial consensus protein


#include <vector>
#include <string>
using std::vector;
using std::string;


class Protein;



class Consensus
{
public:
  // selects the protein with overall smallest sum of pairwise alignment scores
  static Protein* center(const vector<Protein*>& M);

  // selects the protein with overall smallest maximum pairwise alignment scores
  static Protein* minmax(const vector<Protein*>& M);

  // selects the protein that gives the largest initial core
  static Protein* maxcore(const vector<Protein*>& M);
  
  // selects the protein of median length
  static Protein* median(const vector<Protein*>& M);

  // selects the protein with hte given index
  static Protein* index(const vector<Protein*>& M);

  // returns corresponding consensus given one of "ceneter", "minmax", "median"
  static Protein* select(const vector<Protein*>& M,
			 const string& choice);
};


#endif
