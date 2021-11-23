

#ifndef __SUMMARY_H__
#define __SUMMARY_H__


// Functions for saving information about the multiple alignment
// (e.g. NBRF/PIR format, coordinates of the transformed proteins,
// coordinates of the consensus, transformation matrices, etc.)


#include <vector>
#include <string>
using std::vector;
using std::string;


struct Alignment;
class Protein;



// saves the multiple alignment in NBRF/PIR format
void write_pir(const vector<Alignment*>& MA, const vector<Protein*>& M,
	       const string& filename);


// saves the multiple alignment in column-wise format (one column per protein)
// each column lists: #gaps  rmsd  (-1 (gap) or the index of the matched residue)*
void write_alignment(const vector<Alignment*>& MA,
		     const vector<Protein*>& M,
		     const string& filename);


// writes the rotation matrices to the given file
void write_matrices(const vector<Protein*>& M,
		    const string& filename);


// writes alignment statistics:
//    min/max/ave protein length, core size, core rmsd, z-score, e-value, time
//
void write_summary(const vector<Alignment*>& MA,
		   const vector<Protein*>& M,
		   double cpu_time,
		   const string& filename);


// computes the min/max/ave protein length (returned by reference)
bool length_stats(const vector<Protein*>& M,
		  int& outMin,
		  int& outMax,
		  int& outAve);


// writes the original ATOM records for the given proteins but with the
// new (transformed) coordinates that correspond to the computed alignment
// (saved under the original name plus the given suffix (e.g. ".rot"))
void write_coords(const vector<Protein*>& M,
		  const string& suffix);


// writes the coordinates of the given protein to a PDB-formatted file
//     as a sequence of ATOM records of alpha-carbon type (CA),
//     XXX for amino acid type, and consecutive serial and seq. numbers
void write_consensus(Protein* J, const vector<Protein*>& M);


#endif
