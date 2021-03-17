

#include <string>
#include <algorithm>
#include <sstream>
using std::string;
using std::istringstream;


#include "pdbfilter.h"
#include "atomiter.h"
#include "pdbatom.h"



// PDBFilter implementation

PDBFilter::PDBFilter(AtomIter& it) : it(it)
{
}

const PDBAtom& PDBFilter::operator*() const
{
  return *it;
}

const PDBAtom* PDBFilter::operator->() const
{
  return &*it;
}

PDBFilter::operator bool() const
{
  return it;
}



// ModelFilt implementation

ModelFilt::ModelFilt(AtomIter& it) : PDBFilter(it)
{
}

PDBFilter& ModelFilt::operator++()
{
  ++it;
  return *this;
}



// ChainFilt implementation

ChainFilt::ChainFilt(AtomIter& it, char c) : PDBFilter(it), chain(c)
{
  while(it) {
    if (it->getChain() == c) {
      return;
    }
    ++it;
  }
}

ChainFilt::ChainFilt(AtomIter&it, char c, int start) : PDBFilter(it), chain(c)
{
  while(it) {
    if (it->getChain() == c && it->getSeq() == start) {
      return;
    }
    ++it;
  }
}
  
PDBFilter& ChainFilt::operator++()
{
  ++it;
  if (!it || it->getChain() != chain) {
    it = AtomIter();
  }
  return *this;
}



// RangeFilt implementation

RangeFilt::RangeFilt(AtomIter& it,
		     char c1, int s1,
		     char c2, int s2) : ChainFilt(it, c1, s1),
					endChain(c2), endSeq(s2),
					_eor(false)
{
}

PDBFilter& RangeFilt::operator++()
{
  PDBAtom curr = (*it);

  // there could be multiple *last* records when all ATOM records
  // are traversed, as opposed to only alpha-carbons.
  if (isLast(curr)) {
    ++it;
    if (!it || !isLast(*it)) {
      it = AtomIter();
      _eor = true;
    }
  }
  else {    // last record not found, but end of range detected
    ++it;
    if ( !it || (curr.getChain() == endChain &&
		 it->getChain() != endChain) ) {
      it = AtomIter();
      _eor = false;
    }
  }
  
  return *this;
}

bool RangeFilt::eor() const
{
  return _eor;
}

bool RangeFilt::isLast(const PDBAtom& r) const
{
  return (r.getChain() == endChain) && (r.getSeq() == endSeq);
}



// Creating PDB filters given a string description (colon-separated values)

PDBFilter* PDBFilter::create(AtomIter& iter, const string& spec)
{
  istringstream stream(spec);
  
  int tags = std::count(spec.begin(), spec.end(), ':');
  
  if (spec == "") {
    return new ModelFilt(iter);
  }
  else if (tags == 0) {
    char chain;
    stream >> chain;
    if (stream) {
      return new ChainFilt(iter, chain);
    }
  }
  else if (tags == 1) {
    char chain, colon;
    int seq;

    stream >> chain >> colon >> seq;
    if (stream) {
      return new ChainFilt(iter, chain, seq);
    }
  }
  else if (tags == 3) {
    char startChain, stopChain, colon;
    int startSeq, stopSeq;
    stream >> startChain >> colon >> startSeq
	   >> colon
	   >> stopChain >> colon >> stopSeq;
    if (stream) {
      return new RangeFilt(iter, startChain, startSeq, stopChain, stopSeq);
    }
  }

  return 0;
}
