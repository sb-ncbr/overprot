

#ifndef __UTILS_H__
#define __UTILS_H__


#include <string>
#include <map>
using std::string;
using std::map;



// checks if the given string starts with the given prefix
bool startswith(const string& s, const string& prefix);


// returns the portion of the given string after the last '/'
string replace(string s, char oldC, char newC);


// returns the given string with all occurrences of oldC replaced by newC
string basename(const string& path);


// joins the given strings with ":" (used for formatting the output)
string format_range(const string& filename, const string& range);


// extracts the values for the command line options ("-p" and "-n")
// default values are:
//    -p   "./"   -- input directory is current directory
//    -n   "alignment"   -- prefix result files with alignment
//
map<string, string> parse_cmd(int argc, char** argv);


// returns the 1-character amino acid code for the given 3-character abbreviation
char aminoCode(string amino);


#endif
