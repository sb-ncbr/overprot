

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::string;
using std::vector;
using std::ofstream;
using std::fixed;
using std::setprecision;
using std::setw;
using std::istringstream;


#include "protein.h"
#include "point.h"
#include "angles.h"
#include "matrix.h"
#include "utils.h"

#include "atomiter.h"
#include "pdbfilter.h"
#include "msvd.h"



Protein::Protein()
{
}


// computes the angle representation of the protein
// (see Section 3.1, p. 5, of Ye et al. "Pairwise protein structure alignment)
void Protein::compute_angles()
{
  // compute the direction between every pair of atoms
  vector<Point> dirs(this->size()-1);
  for (int i = 0; i < this->size()-1; ++i) {
    Point atom1 = this->atoms[i];
    Point atom2 = this->atoms[i+1];
      
    dirs[i] = atom2 - atom1;
    dirs[i].normalize();
  }

  // compute the angles determined by every triple of directions
  this->angles = vector<AngleTriple>(dirs.size()-2);
  for (int i = 0; i < dirs.size()-2; i++) {
    AngleTriple triple(dirs[i], dirs[i+1], dirs[i+2]);
    this->angles[i] = triple;
  }
}


// returns the number of atoms in the protein
int Protein::size() const
{
  return atoms.size();
}


// transforms the protein coordinates by the given rotation and translation
void Protein::orient(const Matrix<double>& R, const Point& T)
{    
  for (int i = 0; i < this->size(); i++) {
    Point p = this->atoms[i];
    this->atoms[i] = R * p + T;
  }
}


// computes the transformations from original to the current orientation
void Protein::transforms(Matrix<double>& rot, Point& trans)
{
  int n = this->size();
  MSVD::svd(this->orig_atoms, this->atoms, 0, n-1, 0, n-1,
	    rot, trans);
}


// returns a copy of the protein
Protein* Protein::clone() const
{
  Protein* copy = new Protein();
  assert(copy != 0);

  copy->filename = this->filename;

  copy->orig_atoms = this->orig_atoms;
  copy->atoms = this->atoms;
  copy->angles = this->angles;

  return copy;
}


// reads the backbone Ca atoms from the PDB-formatted file with the given path;
// the optional parameter *range* specifies what portion should be read
//    e.g. "" - all CAs of first model
//         "A" - all of chain A only
//         "A:4" - chain A starting at CA 4,
//         "A:4:B:10" - chain A, 4-th CA to chain B, 10th CA
Protein* Protein::read(const string& path, const string& range) 
{
  ifstream in(path.c_str());
  if (in.fail()) {
    cerr << "read error: could not open PDB file " << path << endl;
    exit(1);
  }
  
  AtomIter iter(in);
  PDBFilter* filter  = PDBFilter::create(iter, range);
  if (filter == 0) {
    cerr << "read error: invalid range pattern in " << format_range(path, range) << endl;
    return 0;
  }
  
  Protein* prot = new Protein();
  prot->filename = path;
  prot->range = range;
  
  while ( *filter ) {
    PDBAtom record = **filter;
    Point p = Point(record.getX(), record.getY(), record.getZ());
    prot->atoms.push_back(p);

    ++*filter;
  }
  in.close();


  bool error = false;
  string label = basename(prot->filename) + ":" + prot->range;

  RangeFilt* rangeFilt = dynamic_cast<RangeFilt*>(filter);
  if (rangeFilt && !rangeFilt->eor()) {
    cerr << "read error: could not find the exact range in " << label << endl;
    error = true;
  }
  else if (prot->size() == 0) {
    cerr << "read error: could not find Ca atoms in " << label << endl;
    error = true;
  }
  else if (prot->size() < 4) {
    cerr << "read error: need at least 4 Ca atoms, found " 
	 << prot->size() << " in " << label << endl;
    error = true;
  }
  
  delete filter;
  if (error) {
    delete prot;
    return 0;
  }
  
  prot->orig_atoms = prot->atoms;
  prot->compute_angles();
  
  return prot;
}
  
  

// saves the current protein coordinates to a file under
// the original filename plus the given suffix
void Protein::write(const string& suffix) 
{
  string filename = this->filename + suffix;
  ofstream out(filename.c_str());
  if (out.fail()) {
    cout << "could not save protein data to file " << filename << endl;
    return;
  }
  for (int i = 0; i < this->size(); i++) {
    Point atom = this->atoms[i];
    out << fixed
	<< setprecision (3)
	<< atom.x() << " "
	<< atom.y() << " "
	<< atom.z() << endl;
  }
}



// static methods of class Protein

// convenience method for reading a set of proteins given:
// - the path where the protein data is stored ("./" for current dir)
// - the list of protein files to process
// (see manual for format of data set description file)
// 
vector<Protein*> Protein::read_set(const string& path, const string& data_set)
{
  vector<Protein*> M;
  
  ifstream in(data_set.c_str());
  if (in.fail()) {
    cerr << "data set error: could not open description file " << data_set << endl;
    exit(1);
  }

  bool read_error = false;
  
  string line;
  while(getline(in, line)) {
    istringstream stream(line);
    
    string name, range, tmp;
    stream >> name >> range >> tmp;
    
    if (name == "" || name[0] == '#') {
      continue;    // blank line OR comment line
    }
    
    // check for extraneous information on current line
    if (tmp != "") {
      cerr << endl << "data set error: too many entries on line" << endl;
      cerr << endl << "\t" << line << endl;
      exit(1);
    }
    
    // read a protein and abort on read error
    string prot_path = path + "/" + name;
    Protein* prot = Protein::read(prot_path, range);
    if (prot == 0) {
      read_error = true;
      continue;
    }

    M.push_back(prot);

    // display log of successfully read proteins
    cout << setw(3) << M.size() << ": "
	 << "length - " << setw(4) << M.back()->size() << ", "
	 << "range - " << format_range(M.back()->filename,  M.back()->range)
	 << endl;
  }
  in.close();

  if (read_error) {
    exit(1);
  }
  else if (M.size() < 2) {
    cerr << "data set error: please provide at least two structures" << endl;
  }
    
  return M;
}


// convenience method for writing the (x,y,z) coordinates of a protein
// (saved under the original name plus the given suffix (e.g. ".xyz"))
void Protein::write_set(const vector<Protein*>& M, const string& suffix)
{
  for (int i = 0; i < M.size(); ++i) {
    M[i]->write(suffix);
  }
}


// convenience method for releasing a protein set
void Protein::free(const vector<Protein*>& M)
{
  for (int i = 0; i < M.size(); ++i) {
    delete M[i];
  }
}
