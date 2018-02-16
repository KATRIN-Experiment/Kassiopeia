#include "KConicSection.hh"

#include "KEMConstants.hh"

namespace KEMField
{
  void KConicSection::SetValues(const KPosition& p0,
  				const KPosition& p1)
  {
    fP0 = p0;
    fP1 = p1;
  }

  void KConicSection::SetValues(const double& r0,
  				const double& z0,
  				const double& r1,
  				const double& z1)
  {
    fP0[0] = r0; fP0[1] = 0.; fP0[2] = z0;
    fP1[0] = r1; fP1[1] = 0.; fP1[2] = z1;
  }

  double KConicSection::Area() const
  {
    return KEMConstants::Pi * (fP0[0] + fP1[0])*sqrt((fP0[0] - fP1[0])*(fP0[0] - fP1[0]) + (fP0[2] - fP1[2])*(fP0[2] - fP1[2]));
  }

  double KConicSection::DistanceTo(const KPosition& aPoint,KPosition& nearestPoint)
  {
    double r = sqrt(aPoint[0]*aPoint[0]+aPoint[1]*aPoint[1]);

    double u = ((r-fP0[0])*(fP1[0]-fP0[0]) + (aPoint[2]-fP0[2])*(fP1[2]-fP0[2]))/(fP1-fP0).MagnitudeSquared();

    double cos = aPoint[0]/sqrt(aPoint[0]*aPoint[0]+aPoint[1]*aPoint[1]);
    double sin = aPoint[1]/sqrt(aPoint[0]*aPoint[0]+aPoint[1]*aPoint[1]);

    if (u<=0.)
    {
      nearestPoint[0] = fP0[0]*cos;
      nearestPoint[1] = fP0[0]*sin;
      nearestPoint[2] = fP0[2];
      return sqrt((r-fP0[0])*(r-fP0[0])+(aPoint[2]-fP0[2])*(aPoint[2]-fP0[2]));
    }
    else if (u>=1.)
    {
      nearestPoint[0] = fP1[0]*cos;
      nearestPoint[1] = fP1[0]*sin;
      nearestPoint[2] = fP1[2];
      return sqrt((r-fP1[0])*(r-fP1[0])+(aPoint[2]-fP1[2])*(aPoint[2]-fP1[2]));
    }
    else
    {
      double r_int = fP0[0] + u*(fP1[0]-fP0[0]);
      double z_int = fP0[2] + u*(fP1[2]-fP0[2]);

      nearestPoint[0] = r_int*cos;
      nearestPoint[1] = r_int*sin;
      nearestPoint[2] = z_int;

      return sqrt((r-r_int)*(r-r_int) + (aPoint[2]-z_int)*(aPoint[2]-z_int));
    }
  }

  /**
   * Returns the normal vector to the surface at the point P on
   * the surface.  The normal points outwards if the vector pointing from P0 to
   * P1 is in the first or fourth quadrant in the r-z plane, and points inwards
   * if it is in the second or third quadrant.
   */
  const KDirection KConicSection::Normal() const
  {
    KDirection norm;
    // First, we figure out the unit normal if it were lying in the x-z plane.
    norm[1] = 0.;

    // If the line describing the CS is vertical...
    if (fabs(fP0[2]-fP1[2])<1.e-10)
    {
      norm[0] = 0.;
      // ... and the line is pointed in positive x, the unit normal points in
      // negative z.  Otherwise, it points in positive z.
      if (fP0[0]<fP1[0])
  	norm[2] = -1.;
      else
  	norm[2] = 1.;
    }
    // Otherwise, if the line describing the CS is horizontal...
    else if (fabs(fP0[0]-fP1[0])<1.e-10)
    {
      norm[2] = 0.;
      // ... and the line is pointed in positive z, the unit normal points in
      // positive x.  Otherwise, it points in negative x.
      if (fP0[2]<fP1[2])
  	norm[0] = 1.;
      else
  	norm[0] = -1.;
    }
    // Otherwise, the unit normal is just the negative slope of the generating
    // line.
    else
    {
      norm[0] = ((fP0[2]-fP1[2])/(fP1[0]-fP0[0]))/
  	sqrt(1.+((fP0[2]-fP1[2])/(fP1[0]-fP0[0]))*((fP0[2]-fP1[2])/(fP1[0]-fP0[0])));
      norm[1] = 0.;
      norm[2] = 1./sqrt(1.+((fP0[2]-fP1[2])/(fP1[0]-fP0[0]))*((fP0[2]-fP1[2])/(fP1[0]-fP0[0])));
    }
    return norm;
  }
}
