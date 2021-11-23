

#ifndef __PDB_ATOM_H__
#define __PDB_ATOM_H__


// An abstraction of a PDB ATOM record:
//
// http://www.wwpdb.org/documentation/format32/sect9.html#ATOM
//
// Offers functionality for extracting and replacing the data in certain fields
// of the record (e.g. x-, y-, z-coord, serial and sequence number, etc.)


#include <iosfwd>
#include <string>
using std::string;
using std::ostream;



class PDBAtom
{
public:
  // creates a blank record of size 80
  PDBAtom();

  // creates a record of the given data (must start with "ATOM" and
  // be at least 54 characters long (up to z-coord at least)
  PDBAtom(const string& s);

  // get the string representation of the record
  const string& getData() const;

  
  // selectors for certain fields in the record
  
  string getType() const;

  string getAmino() const;
  
  char getChain() const;

  int getSeq() const;
  
  double getX() const;
  
  double getY() const;
  
  double getZ() const;

  
  // mutators for some of the fields
  
  void setX(double x);

  void setY(double y);

  void setZ(double z);

  void setSeq(int n);

  void setSerial(int n);

  void setType(const string& n);

  void setAmino(const string& n);

  
  // is the record for carbon-alpha atom
  bool isCA () const;

private:
  // formatting helper methods used in the setXX methods

  // returns a string representation of the given value
  // in the given width (w) and precision (p)
  string format(double value, int w, int p);

  // returns a string representation of the given value
  // in the given width (w)
  string format(int value, int w);


private:
  string record;
};


ostream& operator<<(ostream& os, const PDBAtom& r);


#endif
