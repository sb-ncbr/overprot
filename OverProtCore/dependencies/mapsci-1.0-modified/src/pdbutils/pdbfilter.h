

#ifndef __PDB_FILTER_H__
#define __PDB_FILTER_H__


// Abstractions for traversing the ATOM records in a PDB file by filtering
// only those records that match certain criteria. Implements the same
// interface as described in AtomIter (but see RangeFilt below).
//
// The current filters are:
//
// ModelFilt -- traverses all records (from first model)
//
// ChainFilt -- traverses only those records that belong to a given chain
//
// RangeFilt -- traverses only those records that belong to a given range
//              (between a given start chain,position and end chain,position)
//
// Example usage:
//
// ifstream in("1tnr.pdb")
// AtomIter iter(in);
//
// ModleFilt f1(iter);  --  all records
//
// ChainFilt f2(iter, "B");  --  only records of chain B
//
// ChainFilt f3(iter, "B", 10);  --  only records of chain B starting at
//                                   record with sequence number 10
//
// RangeFilt f4(iter, "A", 10, "B", 20);  --  from record with chain A, seq. number 10
//                                            to record with chain B, seq. number 20
//                                            inclusive
//
// A convenience method exists for creating a filter given a string
// description (colon-separated values). The above can be created:
//
// PDBFilter* f1 = PDBFilter::create(iter, "");
//
// PDBFilter* f1 = PDBFilter::create(iter, "B");
//
// PDBFilter* f1 = PDBFilter::create(iter, "B:10");
//
// PDBFilter* f1 = PDBFilter::create(iter, "A:10:B:20");


#include <string>
using std::string;


class AtomIter;
class PDBAtom;



// The base class for PDBFilters. Derived classes must implement prefix ++ .
class PDBFilter
{
public:
  PDBFilter(AtomIter& it);
  
  virtual PDBFilter& operator++() = 0;      // implement in derived classes
  
  const PDBAtom& operator*() const;         

  const PDBAtom* operator->() const;

  operator bool() const;

  // creates a filter given a colon-separated description (see above)
  static PDBFilter* create(AtomIter& iter, const string& spec);

  
protected:
  AtomIter& it;
};
  


// Traverses all records (whole model)

class ModelFilt : public PDBFilter
{
public:
  ModelFilt(AtomIter& it);

  virtual PDBFilter& operator++();
};



// Traverses records of a given chain (optionally
// a start position within the chain can be given)

class ChainFilt : public PDBFilter
{
public:
  ChainFilt(AtomIter& it, char c);

  ChainFilt(AtomIter&it, char c, int start);
  
  virtual PDBFilter& operator++();

  
private:
  char chain;
};



// Traverses records in a given range
// (from given start chain and position to a given end chain and position, inclusive)
//
// 
// In addition to the implicit conversion to bool to detect if there are no
// more records to be read, this filter implements a method "eor()" which
// determines if the traversal ended because end-of-range was detected (i.e.
// the complete range was read) or because end-of-file was reached (perhaps
// incorrectly specified region)

class RangeFilt : public ChainFilt
{
public:
  RangeFilt(AtomIter& it,
	    char c1, int s1,
	    char c2, int s2);
  
  virtual PDBFilter& operator++();

  bool eor() const;
  
private:
  // checks if this is the last record sought
  bool isLast(const PDBAtom& r) const;

  
private:
  char endChain;
  int endSeq;
  bool _eor;
};


#endif
