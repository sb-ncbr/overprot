

#ifndef __ATOM_ITER_H__
#define __ATOM_ITER_H__


// An abstraction for traversing the ATOM records in a PDB file.
// Implements iterator-like interface: prefix++, *, and -> .
// End of traversal can be detected via implicit bool covewrsion.
//
// One can traverse only the ATOM records for alpha-carbons, or all ATOM
// records by specifying an optional parameter at construction time.
//
// Example usage (traverse only alpha-carbon records):
//
// ifstream in("1tnr.pdb")
// AtomIter iter(in);
// while (iter) {
//    cout << (*iter).getChain() << endl;
//    cout << iter->getChain() << endl;
//    ++iter;
// }
//
//
// To traverse all ATOM records construct the iterator as:
//
// AtomIter iter(in, false);


#include <iosfwd>
using std::ifstream;

#include "pdbatom.h"



class AtomIter
{
 public:
  // creates an empty iterator
  AtomIter();
  
  // associates this iterator with the given stream
  // optional second parameter: true - only alha-carbons; false - all ATOM
  AtomIter(ifstream& i, bool ca = true);

  // advances to next ATOM record; stops "END" or "ENDMDL" record or end-of-file
  AtomIter& operator++();
  
  const PDBAtom& operator*() const;
  
  const PDBAtom* operator->() const;
  
  // implicit bool conversion for testing if more records can be read
  operator bool() const;
  
  
private:
  ifstream* in;        // the associated file stream
  bool calpha;         // true - only alpha-carbons; false - all ATOM records
  PDBAtom record;      // current record pointed to by the iterator
};


#endif
