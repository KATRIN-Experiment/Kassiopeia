#include "KGConicSectPortHousing.hh"

#include <cmath>
#include <algorithm>

namespace KGeoBag
{
  KGConicSectPortHousing::KGConicSectPortHousing(double zA,
						 double rA,
						 double zB,
						 double rB)
  {
    fzAMain = zA;
    frAMain = rA;
    fzBMain = zB;
    frBMain = rB;

    if (zA > zB)
    {
      fzAMain = zB;
      frAMain = rB;
      fzBMain = zA;
      frBMain = rA;
    }

    fNumDiscMain = 30;
    fPolyMain = 120;

    fLength = sqrt((fzAMain-fzBMain)*(fzAMain-fzBMain) +
		   (frAMain-frBMain)*(frAMain-frBMain));
  }

  KGConicSectPortHousing::~KGConicSectPortHousing()
  {
    for (unsigned int i=0;i<fPorts.size();i++)
      delete fPorts.at(i);
  }

  KGConicSectPortHousing* KGConicSectPortHousing::Clone() const
  {
    KGConicSectPortHousing* c = new KGConicSectPortHousing();
    c->fzAMain = fzAMain;
    c->frAMain = frAMain;
    c->fzBMain = fzBMain;
    c->frBMain = frBMain;
    c->fLength = fLength;
    c->fNumDiscMain = fNumDiscMain;
    c->fPolyMain = fPolyMain;

    for (unsigned int i=0;i<fPorts.size();i++)
      c->fPorts.push_back(fPorts.at(i)->Clone(c));
    return c;
  }

  void KGConicSectPortHousing::Initialize() const
  {
    for (unsigned int i=0;i<fPorts.size();i++)
      fPorts.at(i)->Initialize();
  }

  void KGConicSectPortHousing::AddParaxialPort(double asub[3],
					       double rsub)
  {
    // Adds a port to the housing.

    KGConicSectPortHousing::Port* p = new KGConicSectPortHousing::ParaxialPort(this,asub,rsub);
    fPorts.push_back(p);
  }

  void KGConicSectPortHousing::AddOrthogonalPort(double asub[3],
						 double rsub)
  {
    // Adds a port to the housing.

    KGConicSectPortHousing::Port* p =
      new KGConicSectPortHousing::OrthogonalPort(this,
						 asub,
						 rsub);
    fPorts.push_back(p);
  }

  void KGConicSectPortHousing::AddPort(KGConicSectPortHousing::Port* port)
  {
    // Adds a port to the housing.

    port->SetPortHousing(this);
    fPorts.push_back(port);
  }

  double KGConicSectPortHousing::GetRAlongConicSect(double z) const
  {
    // Given a z-value, returns the radius of the conic section.

    double m = (frAMain-frBMain)/(fzAMain-fzBMain);
    double b = frAMain - m*fzAMain;
    return m*z + b;
  }

  double KGConicSectPortHousing::GetZAlongConicSect(double r) const
  {
    // Given an r-value, returns the z-component of the conic section.

    double m = (fzAMain-fzBMain)/(frAMain-frBMain);
    double b = fzAMain - m*frAMain;
    return m*r + b;
  }

  void KGConicSectPortHousing::
  RayConicSectIntersection(const std::vector<double>& p0,
			   const std::vector<double>& n1,
			   std::vector<double>& p_int) const
  {
    // In three dimensions, computes the point of intersection between a ray
    // from p0 and pointing in the direction of n1 and the main conic section.
    //  It is assumed that the ray lies in a plane that is parallel to the
    // z-axis.  The method fills p_int with the intersection point.

    // We start by computing the line formed by the intersection of the conic
    // section and a plane parallel to the z axis and containing the ray

    // the plane's offset is just the shortest distance between the ray and the
    // z-axis

    std::vector<double> p1(3);
    for (unsigned int i=0;i<3;i++)
      p1[i] = p0[i] + n1[i];

    std::vector<double> z0(3,0);
    std::vector<double> z1(3,0);
    z1[2] = 1.;

    double m = DistanceBetweenLines(p0,p1,z0,z1);

    // we then compute the point of intersection between the ray and this line
    std::vector<double> projectedP0(2);
    projectedP0[0] = sqrt(p0[0]*p0[0] + p0[1]*p0[1] - m*m);
    if ((p0[0]*n1[0] + p0[1]*n1[1])<0.)
      projectedP0[0]*=-1.;
    projectedP0[1] = p0[2];
    std::vector<double> projectedP1(2);
    projectedP1[0] = sqrt(p1[0]*p1[0] + p1[1]*p1[1] - m*m);
    if ((p1[0]*n1[0] + p1[1]*n1[1])<0.)
      projectedP1[0]*=-1.;
    projectedP1[1] = p1[2];
    std::vector<double> projectedP_int(2);

    // we convert the conic section definition into a generating line r = a*z + b
    double a = ((frBMain-frAMain)/(fzBMain-fzAMain));
    double b = frAMain - a*fzAMain;

    // we convert the ray into a generating line z = g*r + p
    double g = (projectedP1[1]-projectedP0[1])/(projectedP1[0]-projectedP0[0]);
    double p = projectedP0[1] - g*projectedP0[0];

    // we solve for the intersection between the hyperbola and the ray
    double aa,bb,cc;
    aa = g*g*a*a-1.;
    bb = 2.*(g*g*a*b + p);
    cc = g*g*b*b - g*g*m*m - p*p;

    if (fabs(aa)>1.e-10)
    {
      projectedP_int[1] = (-bb-sqrt(bb*bb-4.*aa*cc))/(2.*aa);
      if (projectedP_int[1]<fzAMain ||
	  projectedP_int[1]>fzBMain)
	projectedP_int[1] = (-bb+sqrt(bb*bb-4.*aa*cc))/(2.*aa);
    }
    else
      projectedP_int[1] = -cc/bb;

    projectedP_int[0] = (projectedP_int[1]-p)/g;

    // now we have the point of intersection projected on the plane.  We need only
    // project back

    double d = sqrt((projectedP_int[0]-projectedP0[0])*
		    (projectedP_int[0]-projectedP0[0]) +
		    (projectedP_int[1]-projectedP0[1])*
		    (projectedP_int[1]-projectedP0[1]));

    for (unsigned int i=0;i<3;i++)
      p_int[i] = p0[i] + d*n1[i];

    return;
  }

  double KGConicSectPortHousing::
  DistanceBetweenLines(const std::vector<double>& s1,
		       const std::vector<double>& s2,
		       const std::vector<double>& p1,
		       const std::vector<double>& p2) const
  {
    // In three dimensions, returns the shortest distance between line s (running
    // through s1 and s2) and line p (running through p1 and p2).

    std::vector<double> u(3);
    std::vector<double> v(3);
    std::vector<double> w(3);

    double a = 0;
    double b = 0;
    double c = 0;
    double d = 0;
    double e = 0;

    for (unsigned int i=0;i<3;i++)
    {
      u[i] = s2[i]-s1[i];
      v[i] = p2[i]-p1[i];
      w[i] = s1[i]-p1[i];

      a += u[i]*u[i];
      b += u[i]*v[i];
      c += v[i]*v[i];
      d += u[i]*w[i];
      e += v[i]*w[i];
    }

    double D = a*c - b*b;

    double sc, tc;

    // compute the line parameters of the two closest points
    if (D < 1.e-8)
    {
      // the lines are almost parallel
      sc = 0.0;
      tc = (b>c ? d/b : e/c);   // use the largest denominator
    }
    else
    {
      sc = (b*e - c*d) / D;
      tc = (a*e - b*d) / D;
    }

    std::vector<double> dP(3);

    // get the difference of the two closest points
    for (unsigned int i=0;i<3;i++)
      dP[i] = w[i] + sc*u[i] - tc*v[i];

    // return the closest distance
    return sqrt(dP[0]*dP[0] + dP[1]*dP[1] + dP[2]*dP[2]);
  }

  double KGConicSectPortHousing::DistanceToConicSect(const double* P) const
  {
    // Returns the shortest distance between the conic section and point P.

    double r = sqrt(P[0]*P[0]+P[1]*P[1]);

    double length = sqrt((frAMain-frBMain)*
			 (frAMain-frBMain)+
			 (fzAMain-fzBMain)*
			 (fzAMain-fzBMain));

    double u = ((r-frAMain)*(frBMain-frAMain) + (P[2]-fzAMain)*(fzBMain-fzAMain))/(length*length);

    if (u<=0.)
      return sqrt((r-frAMain)*(r-frAMain) + (P[2]-fzAMain)*(P[2]-fzAMain));
    else if (u>=1.)
      return sqrt((r-frBMain)*(r-frBMain) + (P[2]-fzBMain)*(P[2]-fzBMain));
    else
    {
      double r_int = frAMain + u*(frBMain-frAMain);
      double z_int = fzAMain + u*(fzBMain-fzAMain);

      return sqrt((r-r_int)*(r-r_int) + (P[2]-z_int)*(P[2]-z_int));
    }
  }

  bool KGConicSectPortHousing::ContainedByConicSect(const double* P) const
  {
    // Determines whether or not a point P is in the conic section, without
    // taking into account the ports.

    // if the point is outside of the conic section in the z-direction, return
    // false
    if (P[2]<fzAMain || P[2]>fzBMain)
      return false;

    if (sqrt(P[0]*P[0]+P[1]*P[1])>GetRAlongConicSect(P[2]))
      return false;

    return true;
  }

  bool KGConicSectPortHousing::ContainsPoint(const double* P) const
  {
    if (ContainedByConicSect(P))
      return true;
    else
    {
      // otherwise, we have to look in each of the ports
      for (unsigned int i=0;i<fPorts.size();i++)
	if (fPorts.at(i)->ContainsPoint(P))
	  return true;
    }

    return false;
  }

  double KGConicSectPortHousing::DistanceTo(const double* P,double* P_in,double* P_norm) const
  {
    double P_in_main[3];
    double P_norm_main[3];
    double dist_main = 0.;

    double r = sqrt(P[0]*P[0]+P[1]*P[1]);

    double u = ((r-frAMain)*(frBMain-frAMain) +
		(P[2]-fzAMain)*(fzBMain-fzAMain))/(fLength*fLength);

    double cos=0;
    double sin=0;

    cos = P[0]/sqrt(P[0]*P[0]+P[1]*P[1]);
    sin = P[1]/sqrt(P[0]*P[0]+P[1]*P[1]);

    P_norm_main[0] = (fzBMain - fzAMain)*cos;
    P_norm_main[1] = (fzBMain - fzAMain)*sin;
    P_norm_main[2] = frBMain - frAMain;

    {
      double tmp = 0.;
      for (unsigned int i=0;i<3;i++)
	tmp += P_norm_main[i]*P_norm_main[i];
      for (unsigned int i=0;i<3;i++)
	P_norm_main[i]/=tmp;
    }

    if (u<=0.)
    {
      P_in_main[0] = frAMain*cos;
      P_in_main[1] = frAMain*sin;
      P_in_main[2] = fzAMain;

      dist_main = sqrt((r-frAMain)*(r-frAMain) + (P[2]-fzAMain)*(P[2]-fzAMain));
    }
    else if (u>=1.)
    {
      P_in_main[0] = frBMain*cos;
      P_in_main[1] = frBMain*sin;
      P_in_main[2] = fzBMain;

      dist_main = sqrt((r-frBMain)*(r-frBMain) + (P[2]-fzBMain)*(P[2]-fzBMain));
    }
    else
    {
      double r_int = frAMain + u*(frBMain-frAMain);
      double z_int = fzAMain + u*(fzBMain-fzAMain);

      P_in_main[0] = r_int*cos;
      P_in_main[1] = r_int*sin;
      P_in_main[2] = z_int;

      dist_main = sqrt((r-r_int)*(r-r_int) + (P[2]-z_int)*(P[2]-z_int));

    }
    bool pointIsOnMainConicSect = true;

    for (unsigned int i=0;i<fPorts.size();i++)
    {
      double P_tmp[3];
      double P_tmp_norm[3];
      double dist_tmp = fPorts.at(i)->DistanceTo(P,P_tmp,P_tmp_norm);

      if (dist_tmp<dist_main ||
	  (fPorts.at(i)->ContainsPoint(P_in_main) && pointIsOnMainConicSect))
      {
	pointIsOnMainConicSect = false;

	for (unsigned int j=0;j<3;j++)
	  P_in_main[j] = P_tmp[j];

	for (unsigned int j=0;j<3;j++)
	  P_norm_main[j] = P_tmp_norm[j];

	dist_main = dist_tmp;
      }
    }

    if (P_in)
      for (unsigned int j=0;j<3;j++)
	P_in[j] = P_in_main[j];

    if (P_norm)
      for (unsigned int j=0;j<3;j++)
	P_norm[j] = P_norm_main[j];

    return dist_main;
  }

  //______________________________________________________________________________

  KGConicSectPortHousing::Port::Port(KGConicSectPortHousing* portHousing,double asub[3],double r)
  {
    fPortHousing = portHousing;

    for (int i=0;i<3;i++)
      fASub[i] = asub[i];

    fRSub = r;
  }

  //______________________________________________________________________________

  KGConicSectPortHousing::Port::~Port()
  {
  }

  //______________________________________________________________________________

  KGConicSectPortHousing::OrthogonalPort::OrthogonalPort(KGConicSectPortHousing* pH,double asub[3],double rsub):
    KGConicSectPortHousing::Port(pH,asub,rsub)
  {
    Initialize();
  }

  KGConicSectPortHousing::OrthogonalPort::~OrthogonalPort()
  {
    if (fCoordTransform)
      delete fCoordTransform;
  }

  KGConicSectPortHousing::OrthogonalPort* KGConicSectPortHousing::OrthogonalPort::Clone(KGConicSectPortHousing* c) const
  {
    OrthogonalPort* o = new OrthogonalPort();

    o->fPortHousing = c;
    o->fASub[0] = fASub[0];
    o->fASub[1] = fASub[1];
    o->fASub[2] = fASub[2];
    o->fRSub = fRSub;
    o->fBoxRInner = fBoxRInner;
    o->fBoxROuter = fBoxROuter;
    o->fBoxAngle = fBoxAngle;
    o->fBoxTheta = fBoxTheta;

    o->fLength = fLength;
    o->fAugmentedLength = fAugmentedLength;
    o->fXDisc = fXDisc;
    o->fCylDisc = fCylDisc;
    o->fAlphaPolySub = fAlphaPolySub;
    o->fPolySub = fPolySub;
    o->fSafeHeight = fSafeHeight;

    o->fCoordTransform = new KGCoordinateTransform(*fCoordTransform);

    for (unsigned int i=0;i<3;i++)
    {
      o->fCen[i] = fCen[i];
      o->fX_loc[i] = fX_loc[i];
      o->fY_loc[i] = fY_loc[i];
      o->fZ_loc[i] = fZ_loc[i];
      o->fNorm[i] = fNorm[i];
    }

    return o;
  }

  void KGConicSectPortHousing::OrthogonalPort::Initialize()
  {
    fXDisc = 8;
    fCylDisc = 10;

    fCoordTransform = NULL;

    // we compute the local frame
    ComputeLocalFrame(fCen,fX_loc,fY_loc,fZ_loc);

    fLength = sqrt((fASub[0]-fCen[0])*(fASub[0]-fCen[0]) +
		   (fASub[1]-fCen[1])*(fASub[1]-fCen[1]) +
		   (fASub[2]-fCen[2])*(fASub[2]-fCen[2]));

    fAugmentedLength = sqrt(fASub[0]*fASub[0] +
			    fASub[1]*fASub[1] +
			    fASub[2]*fASub[2])/
      sin(2.*fabs(atan((fPortHousing->GetRBMain()-fPortHousing->GetRAMain())/
		       (fPortHousing->GetZBMain()-fPortHousing->GetZAMain()))));

    fNorm[0] = fCen[0]/sqrt(fCen[0]*fCen[0] + fCen[1]*fCen[1]);
    fNorm[1] = fCen[1]/sqrt(fCen[0]*fCen[0] + fCen[1]*fCen[1]);
    fNorm[2] = 0.;

    fSafeHeight = fRSub*fabs(fX_loc[0]*fNorm[0] + fX_loc[1]*fNorm[1]);

    fCoordTransform = new KGCoordinateTransform(fCen,
						fX_loc,
						fY_loc,
						fZ_loc);
  }

  void KGConicSectPortHousing::OrthogonalPort::ComputeLocalFrame(double *cen,double *x,double *y,double *z)
  {
    // This function computes the local center (0,0,0), x (1,0,0), y (0,1,0) and
    // z (0,0,1) for the port valve (z points up the port to fASub).  This can
    // be done by crossing the normal vector with the z axis, and then crossing
    // that vector with the normal vector.

    double m = -((fPortHousing->GetZAMain()-fPortHousing->GetZBMain())/
		 (fPortHousing->GetRAMain()-fPortHousing->GetRBMain()));

    double theta;
    if (fabs(fASub[0])<1.e-14)
      theta = M_PI/2.;
    else
      theta = atan(fabs(fASub[1]/fASub[0]));
    if (fASub[0]<-0. && fASub[1]>0.)
      theta = M_PI - theta;
    else if (fASub[0]<-0. && fASub[1]<-0.)
      theta += M_PI;
    else if (fASub[0]>0. && fASub[1]<-0.)
      theta = 2.*M_PI - theta;

    z[0] = m*cos(theta);
    z[1] = m*sin(theta);
    z[2] = 1.;

    if (fPortHousing->GetZAMain()<fPortHousing->GetZBMain() &&
	fPortHousing->GetRAMain()<fPortHousing->GetRBMain())
    {
      for (unsigned int i=0;i<3;i++)
	z[i] *= -1.;
    }

    double len = 0;
    for (unsigned int i=0;i<3;i++)
      len+= z[i]*z[i];
    len = sqrt(len);

    for (unsigned int i=0;i<3;i++)
      z[i]/=len;

    double global_z[3] = {0.,0.,1.};

    for (unsigned int i=0;i<3;i++)
      x[i] = z[(i+1)%3]*global_z[(i+2)%3] -
	z[(i+2)%3]*global_z[(i+1)%3];
    for (unsigned int i=0;i<3;i++)
      y[i] = z[(i+1)%3]*x[(i+2)%3] -
	z[(i+2)%3]*x[(i+1)%3];

    double tmp = 0;
    for (unsigned int i=0;i<3;i++)
      tmp += x[i]*x[i];
    tmp = sqrt(tmp);
    for (unsigned int i=0;i<3;i++)
      x[i]/=tmp;

    tmp = 0;
    for (unsigned int i=0;i<3;i++)
      tmp += y[i]*y[i];
    tmp = sqrt(tmp);
    for (unsigned int i=0;i<3;i++)
      y[i]/=tmp;

    std::vector<double> p0(3);
    for (unsigned int i=0;i<3;i++)
      p0[i] = fASub[i];
    std::vector<double> n1(3);
    for (unsigned int i=0;i<3;i++)
      n1[i] = -z[i];
    std::vector<double> p_int(3);

    fPortHousing->RayConicSectIntersection(p0,n1,p_int);

    for (unsigned int i=0;i<3;i++)
      cen[i] = p_int[i];
  }

  bool KGConicSectPortHousing::OrthogonalPort::ContainsPoint(const double* P) const
  {
    // Determines whether or not the point is contained in the port.

    // first, convert to local coordinates
    double P_loc[3];
    fCoordTransform->ConvertToLocalCoords(P,P_loc,false);

    // if the point is outside of the port length, return false
    if (P_loc[2] > fLength)
      return false;

    double r_p = sqrt(P_loc[0]*P_loc[0] + P_loc[1]*P_loc[1]);

    // if the point is outside the port radially, return false
    if (r_p>fRSub)
      return false;

    r_p = sqrt(P[0]*P[0] + P[1]*P[1])+1.e-8;

    double r_cs = fPortHousing->GetRAlongConicSect(P[2]);

    // if the port is inside the main conic section, return false
    if (r_p<r_cs)
      return false;

    // if all of the above conditions are satisfied, return true
    return true;
  }

  double KGConicSectPortHousing::OrthogonalPort::DistanceTo(const double* P,double* P_in,double* P_norm) const
  {
    // Returns the distance between the point <P> and the port.  Additionally,
    // returns the closest point on the port, <P_in>.

    double P_loc[3];
    fCoordTransform->ConvertToLocalCoords(P,P_loc,false);

    // h: the height of the point above (or below) a plane that determines when
    //    the intersection can be ignored
    double h = P_loc[2];

    if (h<0.)
    {
      // if here, we may have to deal with the intersection...

      bool pointIsOnIntersection = true;

      double theta = 0.;
      if (fabs(P_loc[0])>1.e-14)
	theta = atan(fabs(P_loc[1]/P_loc[0]));
      else
	theta = M_PI/2.;

      if (P_loc[1]>0. && P_loc[0]<-0.)
	theta = M_PI - theta;
      else if (P_loc[1]<-0. && P_loc[0]<-0.)
	theta += M_PI;
      else if (P_loc[1]<-0. && P_loc[0]>0.)
	theta = 2.*M_PI - theta;

      std::vector<double> P_tmp_loc(3);
      P_tmp_loc[0] = fRSub*cos(theta);
      P_tmp_loc[1] = fRSub*sin(theta);
      P_tmp_loc[2] = h;

      std::vector<double> P_tmp(3);

      fCoordTransform->ConvertToGlobalCoords(&P_tmp_loc[0],&P_tmp[0],false);

      if (fabs(h)<fSafeHeight)
      {
	// we are in a region that requires further checking

	if (!(fPortHousing->ContainedByConicSect(&P_tmp[0])))
	  pointIsOnIntersection = false;
      }

      if (pointIsOnIntersection)
      {
	// we need to deal with the intersection!
	std::vector<double> n1(3);
	for (unsigned int i=0;i<3;i++)
	  n1[i] = fZ_loc[i];

	std::vector<double> P_int(3);

	fPortHousing->RayConicSectIntersection(P_tmp,n1,P_int);

	for (unsigned int i=0;i<3;i++)
	  P_in[i] = P_int[i];

	{
	  double dot = 0.;
	  for (unsigned int i=0;i<3;i++)
	  {
	    P_norm[i] = P[i] - P_in[i];
	    dot += P_norm[i]*fNorm[i];
	  }
	  for (unsigned int i=0;i<3;i++)
	    P_norm[i] -= dot*fNorm[i];
	}

	return sqrt((P_int[0]-P[0])*(P_int[0]-P[0]) +
		    (P_int[1]-P[1])*(P_int[1]-P[1]) +
		    (P_int[2]-P[2])*(P_int[2]-P[2]));
      }
    }

    std::vector<double> P_int_loc(3);

    // we are above the intersection point, and the problem becomes simply
    // casting to the cylinder
    P_int_loc[2] = (h > fLength) ? fLength : P_loc[2];

    double r_loc = sqrt(P_loc[0]*P_loc[0] + P_loc[1]*P_loc[1]);
    if (r_loc>1.e-14)
    {
      P_int_loc[0] = P_loc[0]*fRSub/r_loc;
      P_int_loc[1] = P_loc[1]*fRSub/r_loc;
    }
    else
      P_int_loc[0] = P_int_loc[1] = 0.;

    fCoordTransform->ConvertToGlobalCoords(&P_int_loc[0],P_in,false);

    {
      double dot = 0.;
      for (unsigned int i=0;i<3;i++)
      {
	P_norm[i] = P[i] - P_in[i];
	dot += P_norm[i]*fNorm[i];
      }
      for (unsigned int i=0;i<3;i++)
	P_norm[i] -= dot*fNorm[i];
    }

    return sqrt((P_int_loc[0]-P_loc[0])*(P_int_loc[0]-P_loc[0]) +
		(P_int_loc[1]-P_loc[1])*(P_int_loc[1]-P_loc[1]) +
		(P_int_loc[2]-P_loc[2])*(P_int_loc[2]-P_loc[2]));
  }


  KGConicSectPortHousing::ParaxialPort::ParaxialPort(KGConicSectPortHousing* portHousing,double asub[3],double rsub) :
    KGConicSectPortHousing::Port(portHousing,asub,rsub)
  {
    Initialize();
  }

  KGConicSectPortHousing::ParaxialPort::~ParaxialPort()
  {
  }

  KGConicSectPortHousing::ParaxialPort* KGConicSectPortHousing::ParaxialPort::Clone(KGConicSectPortHousing* c) const
  {
    ParaxialPort* p = new ParaxialPort();

    p->fPortHousing = c;
    p->fASub[0] = fASub[0];
    p->fASub[1] = fASub[1];
    p->fASub[2] = fASub[2];
    p->fRSub = fRSub;
    p->fBoxRInner = fBoxRInner;
    p->fBoxROuter = fBoxROuter;
    p->fBoxAngle = fBoxAngle;
    p->fBoxTheta = fBoxTheta;

    p->fLength = fLength;
    p->fXDisc = fXDisc;
    p->fCylDisc = fCylDisc;
    p->fAlphaPolySub = fAlphaPolySub;
    p->fPolySub = fPolySub;
    p->fSymmetricLength = fSymmetricLength;
    p->fAsymmetricLength = fAsymmetricLength;
    p->fIsUpstream = fIsUpstream;

    for (unsigned int i=0;i<3;i++)
      p->fNorm[i] = fNorm[i];

    return p;
  }

  void KGConicSectPortHousing::ParaxialPort::Initialize()
  {
    fXDisc = 10;
    fCylDisc = 10;

    // we need to determine in which direction the port opens
    fIsUpstream = (fPortHousing->GetRAMain()<fPortHousing->GetRBMain());
    if (fPortHousing->GetZAMain()>fPortHousing->GetZBMain())
      fIsUpstream = !fIsUpstream;

    ComputeNorm();

    // we can grab the length of the symmetric and asymmetric components of the
    // cylinder by looking at the maximal and minimal points on the intersection

    double z0 = fPortHousing->GetZAMain() -
      ((fPortHousing->GetZBMain()-fPortHousing->GetZAMain())/
       (fPortHousing->GetRBMain()-fPortHousing->GetRAMain()))*
      fPortHousing->GetRAMain();
    double theta =
      2.*atan((fPortHousing->GetRBMain()-fPortHousing->GetRAMain())/
	      (fPortHousing->GetZBMain()-fPortHousing->GetZAMain()));

    fSymmetricLength = fASub[2];
    fAsymmetricLength = 0;

    double zMid = 0;

    for (int i=0;i<2;i++)
    {
      double v = 0.;
      if (fabs(fASub[0])>1.e-10)
	v = atan(fASub[1]/fASub[0]);
      else
      {
	v = M_PI/2.;
	if (fASub[1]<0.)
	  v+=M_PI;
      }

      if (i==1)
	v+=M_PI;

      double t = atan((fRSub*sin(v)+fASub[1])/(fRSub*cos(v)+fASub[0]));
      double u = (fRSub*cos(v)+fASub[0])/(sin(theta/2.)*cos(t));
      double w = (fRSub*cos(v) + fASub[0])/fabs(fRSub*cos(v) + fASub[0])*
	u*cos(theta/2.)+z0;

      if (i==0)
      {
	fAsymmetricLength = w;
	zMid = w;
      }
      else
      {
	fAsymmetricLength-=w;
	zMid += w;
      }
    }

    zMid/=2.;

    fAsymmetricLength = fabs(fAsymmetricLength);
    fSymmetricLength = fabs(fSymmetricLength - zMid) - fAsymmetricLength/2.;
    fLength = fSymmetricLength + .5*fAsymmetricLength;
  }

  void KGConicSectPortHousing::ParaxialPort::ComputeNorm()
  {
    // Computes a unit vector normal to the conic section's surface and coplanar
    // to the axis of the port.

    double m = -((fPortHousing->GetZAMain()-fPortHousing->GetZBMain())/
		 (fPortHousing->GetRAMain()-fPortHousing->GetRBMain()));

    double theta;
    if (fabs(fASub[0])<1.e-14)
      theta = M_PI/2.;
    else
      theta = atan(fabs(fASub[1]/fASub[0]));
    if (fASub[0]<-0. && fASub[1]>0.)
      theta = M_PI - theta;
    else if (fASub[0]<-0. && fASub[1]<-0.)
      theta += M_PI;
    else if (fASub[0]>0. && fASub[1]<-0.)
      theta = 2.*M_PI - theta;

    fNorm[0] = m*cos(theta);
    fNorm[1] = m*sin(theta);
    fNorm[2] = 1.;

    if (fPortHousing->GetZAMain()<fPortHousing->GetZBMain() &&
	fPortHousing->GetRAMain()<fPortHousing->GetRBMain())
      fNorm[2] = -1.;

    double len = 0;
    for (unsigned int i=0;i<3;i++)
      len+= fNorm[i]*fNorm[i];
    len = sqrt(len);

    for (unsigned int i=0;i<3;i++)
      fNorm[i]/=len;
  }

  bool KGConicSectPortHousing::ParaxialPort::ContainsPoint(const double* P) const
  {
    // Determines whether or not the point is contained in the port.

    // first, convert to local coordinates
    double P_loc[3] = {P[0]-fASub[0],P[1]-fASub[1],P[2]-fASub[2]};

    // if the point is outside of the port length, return false
    if ((fIsUpstream && P_loc[2]<0.) ||
	(!fIsUpstream && P_loc[2]>0.))
      return false;

    double r_p = sqrt(P_loc[0]*P_loc[0] + P_loc[1]*P_loc[1]);

    // if the point is outside of the port radius, return false
    if (r_p>fRSub)
      return false;

    r_p = sqrt(P[0]*P[0] + P[1]*P[1])+1.e-8;

    double r_cs = fPortHousing->GetRAlongConicSect(P[2]);

    // if the port is inside the main conic section, return false
    if (r_p<r_cs)
      return false;

    // if all of the above conditions are satisfied, return true
    return true;
  }

  double KGConicSectPortHousing::ParaxialPort::DistanceTo(const double* P,
							  double* P_in,
							  double* P_norm) const
  {
    // Returns the distance between the point <P> and the port.  Additionally,
    // returns the closest point on the port, <P_in>.

    double P_loc[3] = {P[0]-fASub[0],
		       P[1]-fASub[1],
		       P[2]-fASub[2]};

    if ((fIsUpstream && (P_loc[2]>fSymmetricLength)) ||
	((!fIsUpstream) && (P_loc[2]<-fSymmetricLength)))
    {
      // if here, we may have to deal with the intersection...

      bool pointIsOnIntersection = true;

      double theta = 0.;
      if (fabs(P_loc[0])>1.e-14)
	theta = atan(fabs(P_loc[1]/P_loc[0]));
      else
	theta = M_PI/2.;

      if (P_loc[1]>0. && P_loc[0]<-0.)
	theta = M_PI - theta;
      else if (P_loc[1]<-0. && P_loc[0]<-0.)
	theta += M_PI;
      else if (P_loc[1]<-0. && P_loc[0]>0.)
	theta = 2.*M_PI - theta;

      std::vector<double> P_tmp(3);
      P_tmp[0] = fRSub*cos(theta) + fASub[0];
      P_tmp[1] = fRSub*sin(theta) + fASub[1];
      P_tmp[2] = P_loc[2] + fASub[2];

      if (!(fPortHousing->ContainedByConicSect(&P_tmp[0])))
      {
	if ((fIsUpstream && P_loc[2]<(fSymmetricLength + fAsymmetricLength)) ||
	    ((!fIsUpstream) && P_loc[2]>-(fSymmetricLength + fAsymmetricLength)))
	  // the projected point is not in the conic section.  Then we need only
	  // cast to the sheath of the port
	  pointIsOnIntersection = false;
	else
	{
	  // we need to cast the point to the surface of the cone
	  std::vector<double> n1(3);
	  std::vector<double> tmp(3);
	  for (unsigned int i=0;i<3;i++)
	  {
	    P_tmp[i] = P[i];
	    n1[i] = fNorm[i];
	  }
	  fPortHousing->RayConicSectIntersection(P_tmp,n1,tmp);

	  tmp[0]-=fASub[0];
	  tmp[1]-=fASub[1];

	  if (fabs(tmp[0])>1.e-14)
	    theta = atan(fabs(tmp[1]/tmp[0]));
	  else
	    theta = M_PI/2.;

	  if (tmp[1]>0. && tmp[0]<-0.)
	    theta = M_PI - theta;
	  else if (tmp[1]<-0. && tmp[0]<-0.)
	    theta += M_PI;
	  else if (tmp[1]<-0. && tmp[0]>0.)
	    theta = 2.*M_PI - theta;

	  P_tmp[0] = fRSub*cos(theta) + fASub[0];
	  P_tmp[1] = fRSub*sin(theta) + fASub[1];
	  P_tmp[2] = tmp[2];
	}
      }
      else if (!(fPortHousing->ContainedByConicSect(P)))
      {
	// the point is not in the conic section, but it's projection is.  This
	// point must be cast to the intersection
      }
      else
      {
	// we need to cast the point to the surface of the cone
	std::vector<double> n1(3);
	std::vector<double> tmp(3);
	for (unsigned int i=0;i<3;i++)
	{
	  P_tmp[i] = P[i];
	  n1[i] = fNorm[i];
	}
	fPortHousing->RayConicSectIntersection(P_tmp,n1,tmp);

	tmp[0]-=fASub[0];
	tmp[1]-=fASub[1];

	if (fabs(tmp[0])>1.e-14)
	  theta = atan(fabs(tmp[1]/tmp[0]));
	else
	  theta = M_PI/2.;

	if (tmp[1]>0. && tmp[0]<-0.)
	  theta = M_PI - theta;
	else if (tmp[1]<-0. && tmp[0]<-0.)
	  theta += M_PI;
	else if (tmp[1]<-0. && tmp[0]>0.)
	  theta = 2.*M_PI - theta;

	P_tmp[0] = fRSub*cos(theta) + fASub[0];
	P_tmp[1] = fRSub*sin(theta) + fASub[1];
	P_tmp[2] = tmp[2];
      }

      if (pointIsOnIntersection)
      {
	// we need to deal with the intersection!
	std::vector<double> P_int(3);
	P_int[0] = P_tmp[0];
	P_int[1] = P_tmp[1];
	P_int[2] = fPortHousing->GetZAlongConicSect(sqrt(P_tmp[0]*P_tmp[0] +
							 P_tmp[1]*P_tmp[1]));

	for (unsigned int i=0;i<3;i++)
	  P_in[i] = P_int[i];

	{
	  double dot = 0.;
	  for (unsigned int i=0;i<3;i++)
	  {
	    P_norm[i] = P[i] - P_in[i];
	    dot += P_norm[i]*fNorm[i];
	  }
	  for (unsigned int i=0;i<3;i++)
	    P_norm[i] -= dot*fNorm[i];
	}

	return sqrt((P_int[0]-P[0])*(P_int[0]-P[0]) +
		    (P_int[1]-P[1])*(P_int[1]-P[1]) +
		    (P_int[2]-P[2])*(P_int[2]-P[2]));
      }
    }

    std::vector<double> P_int_loc(3);

    // we are above the intersection point, and the problem becomes simply casting
    // to the cylinder
    if (fIsUpstream)
      P_int_loc[2] = (P_loc[2]<0.) ? 0. : P_loc[2];
    else
      P_int_loc[2] = (P_loc[2]>0.) ? 0. : P_loc[2];

    double r_loc = sqrt(P_loc[0]*P_loc[0] + P_loc[1]*P_loc[1]);

    if (r_loc>1.e-14)
    {
      P_int_loc[0] = P_loc[0]*fRSub/r_loc;
      P_int_loc[1] = P_loc[1]*fRSub/r_loc;
    }
    else
      P_int_loc[0] = P_int_loc[1] = 0.;

    P_in[0] = P_int_loc[0] + fASub[0];
    P_in[1] = P_int_loc[1] + fASub[1];
    P_in[2] = P_int_loc[2] + fASub[2];

    {
      double dot = 0.;
      for (unsigned int i=0;i<3;i++)
      {
	P_norm[i] = P[i] - P_in[i];
	dot += P_norm[i]*fNorm[i];
      }
      for (unsigned int i=0;i<3;i++)
	P_norm[i] -= dot*fNorm[i];
    }

    return sqrt((P_int_loc[0]-P_loc[0])*(P_int_loc[0]-P_loc[0]) +
		(P_int_loc[1]-P_loc[1])*(P_int_loc[1]-P_loc[1]) +
		(P_int_loc[2]-P_loc[2])*(P_int_loc[2]-P_loc[2]));
  }

}
