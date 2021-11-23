

#ifndef __ANGLE_TRIPLE_H__
#define __ANGLE_TRIPLE_H__


// An angle triplet describes the angles determined by the planes defined by
// the bonds of 4 consecutive Ca atoms along a protein's backbone. The set of
// all triples gives an oritentation-independent representation of the protein.
//
// For details see Section 3.1., p. 5 of
//
//   Ye at al. "Pairwise protein structure alignment", JBCB, 2 (4) 2004.

#include <ostream>


class Point;



class AngleTriple
{
public:
  // constructs the triplet (0, 0, 0)
  AngleTriple();

  AngleTriple(double a, double b, double g);
  
  // constructs an angle triplet with alpha,beta in [0..PI] and gamma in [0..2*PI]
  // for details see Section 3.1, p. 5 of Ye et al., JBCB, 2 (4) 2004. 
  AngleTriple(const Point& d0, const Point& d1, const Point& d2);

  
  // distance between angle triplets; the adjustment for gamma
  // accounts for the fact that 2 and 358 degrees are close but
  // a direct subtraction will show a large difference
  // for details see Section 3.2, p. 7 of Ye et al., JBCB, 2 (4) 2004. 
  static double distance(const AngleTriple& p, const AngleTriple& q);

  
public:
  double alpha;
  double beta;
  double gamma;
};


std::ostream& operator<<(std::ostream& os, const AngleTriple& angle);

#endif
