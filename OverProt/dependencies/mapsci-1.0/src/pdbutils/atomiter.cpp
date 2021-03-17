

#include "atomiter.h"


#include <fstream>
#include <string>
using std::string;


#include "utils.h"


AtomIter::AtomIter() : in(0)
{
}


AtomIter::AtomIter(ifstream& i, bool ca) : in(&i), calpha(ca)
{
  // positions the iterator at the first ATOM record
  string line;
  while(getline(*in, line)) {
    if (startswith(line, "ATOM")) {
      record = PDBAtom(line);
      if (!calpha || (calpha && record.isCA())) {
	return;
      }
    }
  }
  in = 0;
}


AtomIter& AtomIter::operator++()
{
  // looks for the next ATOM record (stops at END or ENDMDL or end-of-file)
  string s;
  while(getline(*in, s)) {
    if (startswith(s, "END") || startswith(s, "ENDMDL")) {
      in = 0;
      return *this;
    }
    else if (startswith(s, "ATOM")) {
	PDBAtom next(s);
	if (calpha && (next.getSeq() == record.getSeq())) {  // skip duplicates
	  continue;
	}
	else if (calpha && !next.isCA()) {
	  continue;
	}
	
	record = next;
	return *this;
    }
  }
  in = 0;
  return *this;
}


const PDBAtom& AtomIter::operator*() const
{
  return record;
}


const PDBAtom* AtomIter::operator->() const
{
  return &record;
}


AtomIter::operator bool() const
{
  return in && in->is_open();
}
