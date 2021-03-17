

#include <assert.h>
#include <cstdlib>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
using std::string;
using std::ostream;
using std::ifstream;
using std::ostringstream;
using std::ios;
using std::setiosflags;
using std::setprecision;
using std::setw;


#include "pdbatom.h"
#include "utils.h"



PDBAtom::PDBAtom() : record(80, ' ')
{
  record.replace(0, 4, "ATOM");
}


PDBAtom::PDBAtom(const string& s) : record(s)
{
  assert(startswith(record, "ATOM"));
  assert(record.size() >= 54);         // at least up to z-coord field
}


const string& PDBAtom::getData() const
{
  return record;
}
  
string PDBAtom::getType() const
{
  return record.substr(12, 4);
}

string PDBAtom::getAmino() const
{
  return record.substr(17, 3);
}
  
int PDBAtom::getSeq() const
{
  return atoi(record.substr(22, 4).c_str());
}
  
char PDBAtom::getChain() const
{
  return record[21];
}
  
double PDBAtom::getX() const
{
  return atof(record.substr(30, 8).c_str());
}
  
double PDBAtom::getY() const
{
  return atof(record.substr(38, 8).c_str());
}
  
double PDBAtom::getZ() const
{
  return atof(record.substr(46, 8).c_str());
}


void PDBAtom::setX(double x)
{
  string s = format(x, 8, 3);
  record.replace(30, 8, s);
}

void PDBAtom::setY(double y)
{
  string s = format(y, 8, 3);
  record.replace(38, 8, s);
}

void PDBAtom::setZ(double z)
{
  string s = format(z, 8, 3);
  record.replace(46, 8, s);
}

void PDBAtom::setSeq(int n)
{
  string s = format(n, 4);
  record.replace(22, 4, s);
}

void PDBAtom::setSerial(int n)
{
  string s = format(n, 5);
  record.replace(6, 5, s);
}

void PDBAtom::setType(const string& type)
{
  record.replace(12, 4, type);
}

void PDBAtom::setAmino(const string& type)
{
  record.replace(17, 3, type);
}


bool PDBAtom::isCA () const
{
  // note: "CA  " is for calcium atoms, " CA " is for alpha carbon
  return getType() == " CA ";
}


string PDBAtom::format(double value, int w, int p)
{
  ostringstream os;
  
  os << setiosflags(ios::fixed)
     << setprecision(p)
     << setw(w) 
     << value;
    
  return os.str();
}

string PDBAtom::format(int value, int w)
{
  ostringstream os;
  
  os << setw(w) << value;
  
  return os.str();
}


ostream& operator<<(ostream& os, const PDBAtom& r)
{
  os << r.getData();
  return os;
}
