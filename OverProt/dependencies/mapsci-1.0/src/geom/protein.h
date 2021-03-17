

#ifndef __PROTEIN_H__
#define __PROTEIN_H__


// Representation of protein structures as a collection of backbone Ca atoms.
// A protein maintains the original coordinates, the current/transformed
// coordinates and the orientation independent angle-triplets respresentation.
// Every 4 Ca atoms are represented by one angle triplet -- see class AngleTriple.


#include <string>
#include <vector>
using std::string;
using std::vector;


#include "point.h"
#include "angles.h"

template <class T>
class Matrix;



class Protein
{  
public:

  Protein();
  
  
  // computes the angle representation of the protein
  // (see Section 3.1, p. 5, of Ye et al. "Pairwise protein structure alignment)
  void compute_angles();


  // returns the number of atoms in the protein
  int size() const;

    
  // transforms the protein coordinates by the given rotation and translation
  void orient(const Matrix<double>& rot, const Point& trans);

  
  // computes the transformations from original to the current orientation
  void transforms(Matrix<double>& rot, Point& trans);
  

  // returns a copy of the protein (in place of copy c-tor and assign operator)
  Protein* clone() const;


  // reads the backbone Ca atoms from the PDB-formatted file with the given path;
  // the optional parameter *range* specifies what portion should be read:
  //          "" - all CAs of first model
  //         "A" - all of chain A only
  //       "A:4" - chain A starting at CA 4,
  //  "A:4:B:10" - chain A, 4-th CA to chain B, 10th CA (inclusive)
  static Protein* read(const string& path, const string& range = "");
  
    
  // saves the current protein coordinates to a file under
  // the original filename plus the given suffix
  void write(const string& suffix);
  

  // convenience method for reading a set of proteins given:
  // - the path where the protein data is stored ("./" for current dir)
  // - the list of protein files to process
  // (see manual for format of data set description file)
  static vector<Protein*> read_set(const string& path, const string& data_set);


  // convenience method for writing a set of proteins under
  // their original file names plus the given suffix
  static void write_set(const vector<Protein*>& M, const string& suffix);

  
  // convenience method for releasing a protein set
  static void free(const vector<Protein*>& M);

  
private:
  // made private to avoid accidental use (expensive operations)
  Protein(const Protein& orig);
  Protein& operator=(const Protein& orig);


public:
  string filename;                 // source file for protein
  string range;                    // portion read from source file

  vector<Point>  orig_atoms;       // original backbone coordinates
  vector<Point>  atoms;            // current (transformed) coordinates

  vector<AngleTriple>  angles;     // the angle representation (orientation independent)
};


#endif
