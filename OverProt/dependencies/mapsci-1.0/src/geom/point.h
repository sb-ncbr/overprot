

#ifndef __POINT_H__
#define __POINT_H__


// Representation of 3D point/vector with associatred vector operations
// (e.g. dot product, cross product, normalizing, scalar multiplication, etc.)


class Point
{
public:
  Point();   // creates (0, 0, 0)
  
  Point(double x, double y, double z);
  
  double x() const;
  double y() const;
  double z() const;
  double operator[](int i) const;     // access coordinate by index [0 .. 2]

  double length() const;
  void normalize();

  Point& operator+=(const Point& p);
  Point& operator-=(const Point& p);
  Point& operator/=(double factor);

  Point operator+(const Point& p) const;
  Point operator-(const Point& p) const;
  Point operator-() const;
  double operator*(const Point& p) const;     // dot product

  
  // static methods of class point

  static double squaredDistance(const Point& p, const Point& q);
  static double distance(const Point& p, const Point& q);
  static Point cross(const Point& p, const Point& q);       // cross product

  
private:
  double _x, _y, _z;
};


#endif
