

#include <cmath>
#include <assert.h>

#include "point.h"


Point::Point() : _x(0), _y(0), _z(0)
{
}


Point::Point(double xx, double yy, double zz) : _x(xx), _y(yy), _z(zz)
{
}

  
double Point::x() const { return _x; }
double Point::y() const { return _y; }
double Point::z() const { return _z; }

double Point::operator[](int i) const
{
  assert(i >= 0 && i < 3);
  
  double coords[] = {_x, _y, _z};
  return coords[i];
}

double Point::length() const
{
  double len = sqrt(x()*x() + y()*y() + z()*z());
  return len;
}
  
void Point::normalize()
{
  double len = length();
  if (len != 0) {
    (*this) /= len;
  }
}

Point& Point::operator+=(const Point& p)
{
  _x += p.x();
  _y += p.y();
  _z += p.z();
  
  return *this;
}

Point& Point::operator-=(const Point& p)
{
  _x -= p.x();
  _y -= p.y();
  _z -= p.z();
  
  return *this;
}

Point& Point::operator/=(double factor)
{
  assert(factor != 0);
  
  _x /= factor;
  _y /= factor;
  _z /= factor;
  
  return *this;
}

Point Point::operator+(const Point& p) const
{
  double dx = this->x() + p.x();
  double dy = this->y() + p.y();
  double dz = this->z() + p.z();
  
  return Point(dx, dy, dz);
}

Point Point::operator-(const Point& p) const
{
  double dx = this->x() - p.x();
  double dy = this->y() - p.y();
  double dz = this->z() - p.z();
  
  return Point(dx, dy, dz);
}

Point Point::operator-() const
{
  return Point(-x(), -y(), -z());
}

double Point::operator*(const Point& p) const
{
  double dx = this->x() * p.x();
  double dy = this->y() * p.y();
  double dz = this->z() * p.z();
  
  return dx + dy + dz;
}


// static methods of class point

double Point::squaredDistance(const Point& p, const Point& q)
{
  double dx = p.x() - q.x();
  double dy = p.y() - q.y();
  double dz = p.z() - q.z();
  
  double sqDist = dx*dx + dy*dy + dz*dz;
  return sqDist;
}


double Point::distance(const Point& p, const Point& q)
{
  double dist = sqrt(squaredDistance(p, q));
  return dist;
}


Point Point::cross(const Point& p, const Point& q)
{
  double x = p.y()*q.z() - p.z()*q.y();
  double y = - (p.x()*q.z() - p.z()*q.x());
  double z = p.x()*q.y() - p.y()*q.x();
  
  return Point(x, y, z);
}
