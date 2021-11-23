

#include <iostream>
#include <fstream>
#include <string>
using std::cout;
using std::endl;
using std::ifstream;
using std::string;


#include "params.h"



double Params::GAP_PENALTY = 16.0;
double Params::GAP_OPEN    = 4.0;
double Params::GAP_EXT     = 0.2;
double Params::K           = 1.5;
double Params::RUNS_ROT    = 1.0;

double Params::RHO         = 0.0001;   
int Params::MAX_ITERS      = 100;



// load the parameters from file which contains
// one line per parameter value, e.g.:
//    GAP_OPEN 6.0
//    GAP_EXT 0.1
void Params::load(const string& filename)
{
  ifstream in(filename.c_str());
  if (in.fail()) {
    cout << "params_load error: could not open source file" << filename << endl;
    cout << "will use default values:" << endl;

    print();
    
    return;
  }

  string param;
  double value;
  while (in >> param >> value) {
    if (param == "GAP_PENALTY") {
      GAP_PENALTY = value;
    }
    else if (param == "GAP_OPEN") {
      GAP_OPEN = value;
    }
    else if (param == "GAP_EXT") {
      GAP_EXT = value;
    }
    else if (param == "K") {
      K = value;
    }
    else if (param == "RUNS_ROT") {
      RUNS_ROT = value;
    }
    else if (param == "RHO") {
      RHO = value;
    }
    else if (param == "MAX_ITERS") {
      MAX_ITERS = int(value);
    }
  }

  print();
}


// displays the current parameter values
void Params::print()
{
  cout << "using parameters:" << endl;
  cout << "GAP_PENALTY = " << GAP_PENALTY;
  cout << "GAP_OPEN = " << GAP_OPEN;
  cout << "GAP_EXT = " << GAP_EXT;
  cout << "K = " << K;
  cout << "RUNS_ROT = " << RUNS_ROT;
  
  cout << "RHO = " << RHO;
  cout << "MAX_ITERS = " << MAX_ITERS;
}
