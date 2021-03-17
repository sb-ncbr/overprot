
#define _USE_MATH_DEFINES    // for M_PI in MS Visual Studio (2008)
#include <cmath>
#include <assert.h>

#include <ostream>

#include "angles.h"
#include "point.h"


AngleTriple::AngleTriple() : alpha(0), beta(0), gamma(0)
{
}


AngleTriple::AngleTriple(double a, double b, double g) : alpha(a), beta(b), gamma(g)
{
}


// constructs an angle triplet with alpha,beta in [0..PI] and gamma in [0..2*PI]
// for details see Section 3.1, p. 5 of Ye et al., JBCB, 2 (4) 2004. 
AngleTriple::AngleTriple(const Point& d0, const Point& d1, const Point& d2)
{
  // given directions d0,d1 -->*--> the bond angle is <--*-->,
  double ca = -(d0*d1);
  assert(ca >= -1 && ca <= 1);
  alpha = acos(ca);
  
  // given directions d1,d2 -->*--> the bond angle is <--*-->,
  double cb = -(d1*d2);
  assert(cb >= -1 && cb <= 1);
  beta = acos(cb);
  
  // the normals of the planes defined by virtual bonds d0,d1 and d1,d2
  Point n1 = Point::cross(d0, d1);
  Point n2 = Point::cross(d1, d2);
  
  // if any two directions collinear, treat angle between planes (gamma) as 0
  if (n1.length() == 0 || n2.length() == 0) {
    gamma = 0;
  }
  else {
    n1.normalize();
    n2.normalize();
    
    double cg = n1*n2;
    assert(cg >= -1 && cg <= 1);
    

    // n is collinear with d1 (cn is +1 or -1)
    Point n = Point::cross(n2, n1);
    double cn = n*d1;
      
    // compute the third angle, adjusting for direction
    if (cn > 0) {
      gamma = acos(cg);
    }
    else {
      gamma = 2*M_PI - acos(cg);
    }
  }
}


// distance between angle triplets; the adjustment for gamma
// accounts for the fact that 2 and 358 degrees are close but
// a direct subtraction will show a large difference
// for details see Section 3.2, p. 7 of Ye et al., JBCB, 2 (4) 2004. 
double AngleTriple::distance(const AngleTriple& p, const AngleTriple& q)
{
  double da = p.alpha - q.alpha;
  double db = p.beta - q.beta;
  double dg = fabs(p.gamma - q.gamma);
  if (dg > M_PI ) {
    dg = 2*M_PI - dg;
  }
    
  double dis = sqrt(da*da + db*db + dg*dg);
  return dis;
}


std::ostream& operator<<(std::ostream& os, const AngleTriple& angle)
{
  os << "(" << angle.alpha << ", " << angle.beta << ", " << angle.gamma << ")";
  return os;
}
