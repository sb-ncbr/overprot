

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;
using std::left;
using std::right;
using std::setfill;
using std::setw;
using std::ofstream;
using std::vector;


#include "summary.h"
#include "alignment.h"
#include "protein.h"
#include "matrix.h"
#include "atomiter.h"
#include "pairwise.h"
#include "pdbfilter.h"
#include "zscore.h"
#include "utils.h"

// Added by @Adam
void print_matrix(double * matrix, int n_rows, int n_columns, const string filename){
  ofstream out(filename.c_str());
  if (out.fail()) {
    cout << "print_matrix: could not open destination file " << filename
	 << endl;
    return;
  }
  out << "#dtype=float64" << endl;
  out << "#shape=" << n_rows << "," << n_columns << endl;
  for (int j = 0; j < n_columns; ++j) {
    out << "\t" << j;
  }
  out << endl;
  for (int i = 0; i < n_rows; ++i) {
    out << i;
    for (int j = 0; j < n_columns; ++j) {
      out << "\t" << matrix[i*n_columns + j];
    }
    out << endl;
  }
}

// Added by @Adam
void writeQscores(vector<Protein*>& M){
  GlobalAlign global;
  PairAlign pair;
  Alignment* align = new Alignment(100);
  int n = M.size();
  double q_score_matrix[n*n]; 
  for (int i = 0; i < n; ++i) {
    q_score_matrix[i*n + i] = 1.0;
    for (int j = i+1; j < n; ++j) {
      global.align(M[i]->atoms, M[j]->atoms, Point::squaredDistance, align);
      // pair.align(M[i], M[j], align);
      double rmsd = align->computeRMSD(M[i], M[j]); //
      int N_ali = align->matches; //
      int N_i = M[i]->size(); //
      int N_j = M[j]->size(); //
      const double R0 = 3.0; //
      double q_score = 1.0 * N_ali * N_ali / ( N_i * N_j * (1 + (rmsd/R0)*(rmsd/R0)) ); //
      cout << "N_" << i << ", N_" << j << ", RMSD,  N_ali, Q-score: " << N_i << " " << N_j << " " << rmsd << " " << N_ali << " " << q_score << endl; //
      q_score_matrix[i*n + j] = q_score;
      q_score_matrix[j*n + i] = q_score;
    }
  }
  print_matrix(q_score_matrix, n, n, "q_score_matrix.tsv");
}


// saves the multiple alignment in NBRF/PIR format
void write_pir(const vector<Alignment*>& MA, const vector<Protein*>& M,
	       const string& filename)
{
  ofstream out(filename.c_str());
  if (out.fail()) {
    cout << "write_pir error: could not open destination file " << filename
	 << endl;
    return;
  }
  
  for (int i = 0; i < M.size(); ++i) {
    ifstream in(M[i]->filename.c_str());
    if (in.fail()) {
      cout << "write_pir error: could not open PDB file " << M[i]->filename
	   << endl;
      return;
    }
    
    string protname = basename(M[i]->filename);
    
    AtomIter iter(in);
    PDBFilter* filter  = PDBFilter::create(iter, M[i]->range);

    // writes header line: >P1;<prot name>:<range>
    string label = format_range(M[i]->filename, M[i]->range);
    out << ">P1;" << label << endl
	<< label << endl;

    // writes the matched single letter amino acid codes (or -1 for gap)
    // keeps each line to 75 characters; ends info of each protein with *
    for (int j = 0; j < MA[i]->size; ++j) {
      int index = MA[i]->first[j];
      
      char code = '-';
      if (index != GAP_INDEX) {
	code = aminoCode((*filter)->getAmino());
	++*filter;
      }
      
      out << code;
      if ( (j+1) % 75 == 0) {
	out << endl;
      }
    }
    out << '*' << endl;

    delete filter;
  }
}


// saves the multiple alignment in column-wise format (one column per protein)
// each column lists: #gaps  rmsd  (-1 (gap) or the index of the matched residue)*
//
void write_alignment(const vector<Alignment*>& MA,
		     const vector<Protein*>& M,
		     const string& filename)
{
  ofstream out(filename.c_str());
  if (out.fail()) {
    cout << "write_align error: could not open destination file" << filename 
	 << endl;
    return;
  }

  // +2 columns for number of gaps and rmsd in the beginning
  out << "rows  " << MA[0]->size << "  cols  " << M.size()+2 << "  prots  " << M.size() << endl;

  // first row has the protein names + headers for gaps/rmsd
  out << "gaps  rmsd  ";
  for (int i = 0; i < M.size(); ++i) {
    out << format_range(M[i]->filename, M[i]->range)
	<< "  ";
  }
  out << endl;

  // compute #gaps and rmsd
  for (int j = 0; j < MA[0]->size; ++j) {
    // find centroid (cx, cy, cz) of current column
    int matches = 0;
    Point cen(0, 0, 0);
    for (int i = 0; i < M.size(); ++i) {
      int k = MA[i]->first[j];
      if (k != GAP_INDEX) {
	Point atom = M[i]->atoms[k];
	cen += atom;
	
	++matches;
      }
    }
    if (matches != 0) {
      cen /= matches;
      
      // calculate the squared distance to centroid
      // for all atoms in the current column
      double rmsd = 0;
      for (int i = 0; i < M.size(); ++i) {
	int k = MA[i]->first[j];
	if (k != GAP_INDEX) {
	  Point atom = M[i]->atoms[k];
	  rmsd += Point::squaredDistance(cen, atom);
	}
      }

      int gaps = M.size() - matches;
      rmsd = sqrt(rmsd / matches);
      out << gaps << "  " << rmsd;
    }
    else {
      int gaps = M.size() - matches;
      out << gaps << "  " << "-1";
    }

    out << "  ";


    // display the alignment indices for current colum
    for (int i = 0; i < MA.size(); ++i) {
      int k = MA[i]->first[j];
      if (k != GAP_INDEX ) {
	out <<  k << "  ";
      }  
      else{
	out << -1 << "  ";
      } 
    } 
    out << endl;
  }
}


// writes the rotation matrices to the given file
void write_matrices(const vector<Protein*>& M,
		    const string& filename)
{
  ofstream out(filename.c_str());
  if (out.fail()) {
    cout << "write_matrices error: could not open destination file" << filename
	 << endl;
    return;
  }


  out << "# Transformation Matrices                      " << endl
      << "#                                              " << endl
      << "# | x' |   | R00 R01 R02 |   | x |     | T0 |  " << endl
      << "# | y' | = | R10 R11 R12 | * | y |  +  | T1 |  " << endl
      << "# | z' |   | R20 R21 R22 |   | z |     | T2 |  " << endl
      << "#                                              " << endl
      << "#                          R00       R01       R02       R10       R11       R12       R20       R21       R22        T0        T1        T2" << endl
      << "#                                              " << endl;
    
  for (int i = 0; i < M.size(); ++i) {
    // find the optimal transformations that map original coords to new ones
    Matrix<double> rot;
    Point trans;
    M[i]->transforms(rot, trans);

    out << left << setw(20) << setfill(' ')
	<< format_range(M[i]->filename, M[i]->range);

    // print the transposed matrix (for right multiplication)
    out << right << fixed << setprecision(5) << setfill(' ');
    for (int c = 0; c < rot.cols(); ++c) {
      for (int r = 0; r < rot.rows(); ++r) {
	out << setw(10) << rot(r, c);
      }
    }

    out << setprecision(2)
	<< setw(10) << trans.x()
	<< setw(10) << trans.y()
	<< setw(10) << trans.z()
	<< endl;
  }
}


// writes alignment statistics:
//    min/max/ave protein length, core size, core rmsd, z-score, e-value, time
//
void write_summary(const vector<Alignment*>& MA,
		   const vector<Protein*>&   M,
		   double cpu_time,
		   const string& filename)
{
  int minCA = -1, maxCA = -1, aveCA = -1;
  length_stats(M, minCA, maxCA, aveCA);
  
  double coreRmsd = -1;
  int coreCA = -1;
  compute_core(MA, M, 4.0, coreRmsd, coreCA);
  
  double zscore, lnP;
  compute_zscore(coreCA, minCA, zscore, lnP);


  // used for string formatting with sprintf; giving up on iostream formatting
  char buff[256]; 
  

  // completes the output on the screen
  cout << "minCA  maxCA  aveCA  coreCA   RMSD   Z-scr  -ln(P)  Time(s)" << endl; 
  sprintf(buff, "%5d  %5d  %5d  %6d  %5.2f  %6.2f  %5.2f %7.2f", 
	  minCA, maxCA, aveCA, coreCA, coreRmsd, zscore, lnP, cpu_time);
  cout << buff << endl;   
  
  
  // what was shown on the screen is also saved to a file
  ofstream out(filename.c_str());
  if (out.fail()) {
    cout << "write_summary: could not open destination file" << filename
	 << endl;
    return;
  }
  out << endl << "MAPSCI summary for multiple structure alignment of" << endl << endl;
  for (int i = 0; i < M.size(); ++i) {
    sprintf(buff, "%3d: length - %4d, range - %s",
	    i+1, M[i]->size(), format_range(M[i]->filename, M[i]->range).c_str());
    out << buff << endl;
  }
  out << endl;
  out << "minCA  maxCA  aveCA  coreCA   RMSD   Z-scr  -ln(P)  Time(s)" << endl;
  sprintf(buff, "%5d  %5d  %5d  %6d  %5.2f  %6.2f  %5.2f %7.2f", 
	  minCA, maxCA, aveCA, coreCA, coreRmsd, zscore, lnP, cpu_time);
  out << buff << endl;
}
 
 
// computes the min/max/ave protein length (returned by reference)
bool length_stats(const vector<Protein*>& M,
		  int& outMin,
		  int& outMax,
		  int& outAve)
{
  if (M.empty()) {
    cout << "length_stats: no proteins given" << endl;
    return false;
  }
  
  int min_len = M[0]->size();
  int max_len = 0;
  int ave_len = 0;
  for (int i = 0; i < M.size(); ++i) {
    ave_len = ave_len + M[i]->size();
    min_len = std::min(min_len, M[i]->size());
    max_len = std::max(max_len, M[i]->size());
  }
  ave_len  = ave_len/M.size();
  
  outMin = min_len;
  outMax = max_len;
  outAve = ave_len;

  return true;
}


// writes the original ATOM records for the given proteins but with the
// new (transformed) coordinates that correspond to the computed alignment
// (saved under the original name plus the given suffix (e.g. ".rot"))
void write_coords(const vector<Protein*>& M, const string& suffix)
{
  for (int i = 0; i < M.size(); ++i) {
    // find the optimal transformations that map original coords to new ones
    Matrix<double> rot;
    Point trans;
    M[i]->transforms(rot, trans);
    
    
    ifstream in(M[i]->filename.c_str());
    if (in.fail()) {
      cout << "write_coords error: could not open PDB file " << M[i]->filename
	   << endl;
      continue;
    }

    string protname = basename(M[i]->filename);
    if (M[i]->range != "") {
      protname += "." + replace(M[i]->range, ':', '.');
    }
    protname += suffix;

    ofstream out(protname.c_str());
    if (out.fail()) {
      cout << "write_coords error: could not open destination file" << protname
	   << endl;
	continue;
    }

    out << "REMARK" << endl;
    out << "REMARK  Transformed coordinates for " 
	<< format_range(M[i]->filename, M[i]->range) << endl;
    out << "REMARK" << endl;

    AtomIter iter(in, false);
    PDBFilter* filter  = PDBFilter::create(iter, M[i]->range);
    while(*filter) {
      PDBAtom record = **filter;
      
      Point p(record.getX(), record.getY(), record.getZ());
      p = rot * p + trans;
      
      record.setX( p.x() );
      record.setY( p.y() );
      record.setZ( p.z() );
      
      out << record << endl;
      ++*filter;
      
      if ( !*filter || ((*filter)->getChain() != record.getChain()) ) {
	out << "TER" << endl;
      }
    }
    out << "END" << endl;
    
    delete filter;
  }
}


// writes the coordinates of the given protein to a PDB-formatted file
//     as a sequence of ATOM records of alpha-carbon type (CA),
//     XXX for amino acid type, and consecutive serial and seq. numbers
void write_consensus(Protein* J, const vector<Protein*>& M)
{
  ofstream out(J->filename.c_str());
  if (out.fail()) {
    cout << "write_consensus error: could not open destination file"
	 << J->filename << endl;
    return;
  }

  out << "REMARK" << endl;
  out << "REMARK" << endl;
  out << "REMARK  Consensus protein for" << endl;

  out << "REMARK" << endl;
  for (int i = 0; i < M.size(); ++i) {
    out << "REMARK    " << format_range(M[i]->filename, M[i]->range)
	<< endl;
  }
  out << "REMARK" << endl;
  
  PDBAtom record;
  record.setType(" CA ");   // make all alpha-carbon type
  record.setAmino("XXX");   // unknown amino type
  for (int i = 0; i < J->size(); i++) {
    Point p = J->atoms[i];
    
    record.setX( p.x() );
    record.setY( p.y() );
    record.setZ( p.z() );

    record.setSeq(i+1);
    record.setSerial(i+1);
    
    out << record << endl;
  }
  out << "TER" << endl;
  out << "END" << endl;
}
