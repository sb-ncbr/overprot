

#include <cstdlib>

#include <iostream>
#include <string>
#include <utility>
#include <map>
using std::cerr;
using std::endl;
using std::string;
using std::pair;
using std::make_pair;
using std::map;


#include "utils.h"



// checks if the given string starts with the given prefix
bool startswith(const string& s, const string& prefix)
{
  return s.substr(0, prefix.size()) == prefix;
}


// returns the given string with all occurrences of oldC replaced by newC
string replace(string s, char oldC, char newC)
{
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] == oldC) {
      s[i] = newC;
    }
  }
  return s;
}


// returns the portion of the given string after the last '/'
string basename(const string& path)
{
  int i = path.rfind('/');
  return path.substr(i+1, path.size());
}


// joins the given strings with ":" (used for formatting the output)
string format_range(const string& filename, const string& range)
{
  if (range == "") {
    return basename(filename);
  }
  else {
    return basename(filename) + ":" + range;
  }
}


// extracts the values for the command line options ("-p" and "-n")
// default values are:
//    -p   "./"   -- input directory is current directory
//    -n   "alignment"   -- prefix result files with alignment
//
map<string, string> parse_cmd(int argc, char** argv)
{
  map<string, string> params;

  for (int i = 3; i < argc; i += 2) {
    if (params.find(argv[i]) != params.end()) {
      cerr << "option " << argv[i] << " multiply defined" << endl;
      exit(1);
    }
    else if (string(argv[i]) == "-p") {
      if ((i+1) < argc) {
	params["-p"] = argv[i+1];
      }
      else {
	cerr << "option -p should be followed by path to data set files" << endl;
	exit(1);
      }
    }
    else if (string(argv[i]) == "-n") {
      if ((i+1) < argc) {
	params["-n"] = argv[i+1];
      }
      else {
	cerr << "option -n should be followed by label name" << endl;
	exit(1);
      }
    }
    else {
      cerr << "unrecognized command line option " << argv[i] << endl;
      exit(1);
    }
  }
  
  // set default values for command line parameters
  if (params.find("-p") == params.end()) {
    params["-p"] = "./";
  }
  if (params.find("-n") == params.end()) {
    params["-n"] = "alignment";
  }

  return params;
}



// mapping between 3-character amino acid abbreviations and their 1-character code

const pair<string, char> AMINO_TABLE[] = { 
  make_pair("ALA", 'A'),
  make_pair("ARG", 'R'),
  make_pair("ASN", 'N'),
  make_pair("ASP", 'D'),
  make_pair("ASX", 'B'),
  make_pair("CYS", 'C'),
  make_pair("GLN", 'Q'),
  make_pair("GLU", 'E'),
  make_pair("GLX", 'Z'),
  make_pair("GLY", 'G'),
  make_pair("HIS", 'H'),
  make_pair("ILE", 'I'),
  make_pair("LEU", 'L'),
  make_pair("LYS", 'K'),
  make_pair("MET", 'M'),
  make_pair("PHE", 'F'),
  make_pair("PRO", 'P'),
  make_pair("SEC", 'U'),
  make_pair("SER", 'S'),
  make_pair("THR", 'T'),
  make_pair("TRP", 'W'),
  make_pair("TYR", 'Y'),
  make_pair("VAL", 'V'), 
  make_pair("XIE", 'J')
};

const int AMINO_TABLE_SIZE = 24;  // nothing for 'O' and 'X'


// returns the 1-character amino acid code 
// for the given 3-character abbreviation
char aminoCode(string amino)
{
  for (int i = 0; i < amino.size(); ++i) {
    amino[i] = toupper(amino[i]);
  }
    
  for (int i = 0; i < AMINO_TABLE_SIZE; ++i) {
    if (AMINO_TABLE[i].first == amino) {
      return AMINO_TABLE[i].second;
    }
  }
  return 'X';
}
