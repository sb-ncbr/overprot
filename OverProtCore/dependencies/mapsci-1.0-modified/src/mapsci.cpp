
#include <vector>
#include <map>
#include <iostream>
using std::vector;
using std::map;
using std::cout;
using std::cerr;
using std::endl;

#include <stdlib.h>
#include <time.h>

#include "alignment.h"
#include "protein.h"
#include "summary.h"
#include "multiple.h"
#include "utils.h"
#include "msvd.h"



int main(int argc, char** argv)
{
  if (argc < 3) {
    cerr << "usage: " << argv[0] << "  <input set>  <consensus type>  <options>" << endl;
    exit(1);
  } 
  
  char* filenames = argv[1];
  char* consensus = argv[2];

  map<string, string> params = parse_cmd(argc, argv);
  string path = params["-p"];
  string prefix = params["-n"];
  
  
  vector<Protein*> M = Protein::read_set(path, filenames);

  cout << endl << endl;

  cout << "aligning . . ." << endl << endl;
  clock_t start = clock();
  MultiAlign multi;
  MultiAlign::RT result = multi.align(M, consensus);
  clock_t end = clock();
  cout << endl << endl;

  double cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;  

  
  vector<Alignment*> MA = result.first;
  Protein* J = result.second;

  write_summary(MA, M, cpu_time, prefix+".log");
  write_pir(MA, M, prefix+".pir");
  write_matrices(M, prefix+".mat");
  write_alignment(MA, M, prefix+".txt");
  write_coords(M, ".rot");
  write_consensus(J, M);

  writeQscores(M);
  
  Protein::free(M);
  MultiAlign::free(result);
  
  return 0;
}
