#include "KGExtrudedObject.hh"

#include <algorithm>
#include <sstream>

namespace KGeoBag
{
  KGExtrudedObject::~KGExtrudedObject()
  {
    std::vector<KGExtrudedObject::Line*>::iterator it;
    for (it = fInnerSegments.begin();it!=fInnerSegments.end();++it)
      delete * it;
    for (it = fOuterSegments.begin();it!=fOuterSegments.end();++it)
      delete * it;
  }

  KGExtrudedObject* KGExtrudedObject::Clone() const
  {
    KGExtrudedObject* tClone = new KGExtrudedObject();
    tClone->fZMin = fZMin;
    tClone->fZMax = fZMax;
    tClone->fNDisc = fNDisc;
    tClone->fDiscretizationPower = fDiscretizationPower;
    tClone->fNInnerSegments = fNInnerSegments;
    tClone->fNOuterSegments = fNOuterSegments;
    tClone->fClosedLoops = fClosedLoops;

    for(size_t i=0;i<fInnerSegments.size();i++)
      tClone->fInnerSegments.push_back( fInnerSegments[i]->Clone(tClone));

    for(size_t i=0;i<fInnerSegments.size();i++)
      tClone->fInnerSegments.push_back( fInnerSegments[i]->Clone(tClone));

    return tClone;
  }

  void KGExtrudedObject::AddInnerLine(double p1[2],double p2[2])
  {
    // Adds line segment (p1,p2) to inside of the extruded surface.

    fInnerSegments.push_back(new KGExtrudedObject::Line(this,p1,p2));
    fInnerSegments.back()->SetOrder(fNInnerSegments++);
  }

  void KGExtrudedObject::AddOuterLine(double p1[2],double p2[2])
  {
    // Adds line segment (p1,p2) to inside of the extruded surface.

    fOuterSegments.push_back(new KGExtrudedObject::Line(this,p1,p2));
    fOuterSegments.back()->SetOrder(fNOuterSegments++);
  }

  void KGExtrudedObject::AddInnerArc(double p1[2],
				     double p2[2],
				     double radius,
				     bool   positiveOrientation)
  {
    // Adds line segment (p1,p2) to the extruded surface.

    fInnerSegments.push_back(new KGExtrudedObject::Arc(this,p1,p2,radius,positiveOrientation));
    fInnerSegments.back()->SetOrder(fNInnerSegments++);
  }

  void KGExtrudedObject::AddOuterArc(double p1[2],
				     double p2[2],
				     double radius,
				     bool   positiveOrientation)
  {
    // Adds line segment (p1,p2) to the extruded surface.

    fOuterSegments.push_back(new KGExtrudedObject::Arc(this,p1,p2,radius,positiveOrientation));
    fOuterSegments.back()->SetOrder(fNOuterSegments++);
  }

  void KGExtrudedObject::AddInnerSegment(KGExtrudedObject::Line* line)
  {
    // Adds line segment <line> to inside of the extruded surface.

    line->SetExtruded(this);
    fInnerSegments.push_back(line);
    fInnerSegments.back()->SetOrder(fNInnerSegments++);
  }

  void KGExtrudedObject::AddOuterSegment(KGExtrudedObject::Line* line)
  {
    // Adds line segment <line> to inside of the extruded surface.

    line->SetExtruded(this);
    fOuterSegments.push_back(line);
    fOuterSegments.back()->SetOrder(fNOuterSegments++);
  }

  bool KGExtrudedObject::CompareTheta(std::vector<double> p1,
				      std::vector<double> p2)
  {
    // Sorts inner and outer coordinates to run from 0 to 2 Pi

    return (KGExtrudedObject::Theta(p1[0],p1[1]) <
	    KGExtrudedObject::Theta(p2[0],p2[1]));
  }

  bool KGExtrudedObject::RayIntersectsLineSeg(const std::vector<double>& p0,
					      const std::vector<double>& s1,
					      const std::vector<double>& s2,
					      std::vector<double>& p_int)
  {
    // In two dimensions, determines whether or not a ray from the origin and
    // passing through point p0 intersects a line segment defined by points s1
    // and  s2.  If it does not, the method returns false.  If it does, the
    // method returns true and fills p_int with the intersection point.

    double len2;
    double mua,mub;
    double denom,numera,numerb;

    len2   = (s2[0]-s1[0]) * (s2[0]-s1[0]) + (s2[1]-s1[1]) * (s2[1]-s1[1]);
    denom  = (s2[1]-s1[1]) * (p0[0]-0) - (s2[0]-s1[0]) * (p0[1]-0);
    numera = (s2[0]-s1[0]) * (0-s1[1]) - (s2[1]-s1[1]) * (0-s1[0]);
    numerb = (p0[0]-0) * (0-s1[1]) - (p0[1]-0) * (0-s1[0]);

    // Is the line segment a real line segment?
    if (len2 < 1.e-10)
    {
      p_int[0] = 0;
      p_int[1] = 0;
      return false;
    }

    // Are the lines coincident?
    if (fabs(numera) < 1.e-10 && fabs(numerb) < 1.e-10 && fabs(denom) < 1.e-10)
    {
      p_int[0] = p0[0]*.5;
      p_int[1] = p0[1]*.5;
      return true;
    }

    // Are the lines parallel?
    if (fabs(denom) < 1.e-10)
    {
      p_int[0] = 0;
      p_int[1] = 0;
      return false;
    }

    // Is the intersection along the the segment?
    mua = numera / denom;
    mub = numerb / denom;
    if (mua < -1.e-14 || mub < -1.e-14 || mub > 1.)
    {
      p_int[0] = 0;
      p_int[1] = 0;
      return false;
    }

    p_int[0] = mua*p0[0];
    p_int[1] = mua*p0[1];
    return true;
  }

  bool KGExtrudedObject::PointIsInPolygon(std::vector<double> &p1,
					  const std::vector<std::vector<double> > &v,
					  unsigned int vertexStart,
					  unsigned int nVertices)
  {
    // Determines if a point p1 is contained by a polygon with vertices
    // described as the (vertexStart)-th to (vertexStart + nVertices)-th
    // vertices in v. The method used is the Winding Number algorithm described
    // in Alciatore, D. and Miranda, R., "A Winding Number and Point-in-Polygon
    // Algorithm," Glaxo Virtual Anatomy Project research report,  Department of
    // Mechanical Engineering, Colorado State University,  January, 1995.

    double nWindings = 0.;

    std::vector<double> x(nVertices+1);
    std::vector<double> y(nVertices+1);

    for (unsigned int i=0;i<nVertices+1;i++)
    {
      x[i] = v.at(vertexStart+(i%nVertices)).at(0)-p1.at(0);
      y[i] = v.at(vertexStart+(i%nVertices)).at(1)-p1.at(1);
    }

    for (unsigned int i=0;i<nVertices+1;i++)
    {
      if (y[i]*y[i+1]<0.)
      {
	double r = x[i] + ((y[i]*(x[i+1]-x[i]))/(y[i]-y[i+1]));
	if (r>0)
	{
	  if (y[i]<0.) nWindings+=1.;
	  else nWindings-=1.;
	}
      }
      else if (fabs(y[i])<1.e-8 && x[i]>0.)
      {
	if (y[i+1]>0) nWindings+=.5;
	else nWindings-=.5;
      }
      else if (fabs(y[i+1])<1.e-8 && x[i+1]>0.)
      {
	if (y[i]<0) nWindings+=.5;
	else nWindings-=.5;
      }
    }

    if (fabs(nWindings)<.1)
      return false;
    else
      return true;
  }

  double KGExtrudedObject::Theta(const double x,const double y)
  {
    // Returns theta given an x, y in cartesian coordinates.

    double theta;

    if (fabs(x)<1.e-14)
      theta = M_PI/2.;
    else
      theta = atan(fabs(y/x));
    if (x<1.e-14 && y>-1.e-14)
    {
      theta = M_PI - theta;
    }
    else if (x<1.e-14 && y<1.e-14)
      theta += M_PI;
    else if (x>-1.e-14 && y<1.e-14)
      theta = 2.*M_PI - theta;

    return theta;
  }

  bool KGExtrudedObject::ContainsPoint(const double* P) const
  {
    // We tackle this problem in cylindrical coordinates.  First, we discard any
    // points that are not within our z coordinates (since it's easiest).  Then,
    // we test against the radial inner and outer coordinates.  Finally, we
    // compare against and minimal and maximal theta coordinates.  If all of
    // these tests do not reject the point, then the point is in our .

    if (P[2] > fZMin || P[2] > fZMax)
      return false;

    double P_tmp[2];
    double dist_tmp;

    double dist_inner = 1.e6;
    double P_inner[2] = {0,0};

    // we find the closest inner segment to the point
    for (unsigned int i=0;i<fInnerSegments.size();i++)
    {
      dist_tmp = fInnerSegments.at(i)->DistanceTo(P,P_tmp);
      if (dist_tmp<dist_inner)
      {
	dist_inner = dist_tmp;
	P_inner[0] = P_tmp[0];
	P_inner[1] = P_tmp[1];
      }
    }

    double dist_outer = 1.e6;
    double P_outer[2] = {0,0};

    // we find the closest outer segment to the point
    for (unsigned int i=0;i<fOuterSegments.size();i++)
    {
      dist_tmp = fOuterSegments.at(i)->DistanceTo(P,P_tmp);
      if (dist_tmp<dist_outer)
      {
	dist_outer = dist_tmp;
	P_outer[0] = P_tmp[0];
	P_outer[1] = P_tmp[1];
      }
    }

    // assuming both the inner and outer forms wind about the origin (or at
    // least partially do so), we determine the radial and angular values for
    // our point, and the nearest inner and outer points.
    double rad_P = sqrt(P[0]*P[0]+P[1]*P[1]);
    double rad_inner = sqrt(P_inner[0]*P_inner[0] + P_inner[1]*P_inner[1]);
    double rad_outer = sqrt(P_outer[0]*P_outer[0] + P_outer[1]*P_outer[1]);

    if (rad_inner > rad_P || rad_outer < rad_P)
      return false;

    double theta_P = Theta(P[0],P[1]);
    double theta_min = Theta(fInnerSegments.front()->GetP1(0),
			     fInnerSegments.front()->GetP1(1));
    double theta_max = Theta(fInnerSegments.back()->GetP2(0),
			     fInnerSegments.back()->GetP2(1));

    double theta_mid = Theta(fInnerSegments.at(fInnerSegments.size()/2)->
			     GetP2(0),
			     fInnerSegments.at(fInnerSegments.size()/2)->
			     GetP2(1));

    bool orientation = true;
    {
      double thetas[2] = {fmod(theta_mid-theta_min + 2.*M_PI,2.*M_PI),
			  fmod(theta_max-theta_min + 2.*M_PI,2.*M_PI)};
      orientation = (thetas[0] < thetas[1]);
    }

    if (!(Arc::AngleIsWithinRange(theta_P,theta_min,theta_max,orientation)))
      return false;

    return true;
  }

  double KGExtrudedObject::DistanceTo(const double* P,
				      double* P_in,
				      double* P_norm) const
  {
    // First, we determine the nearest z-coordinate
    double z = P[2];
    bool withinZ = false;

    if (P[2] > fZMax) z = fZMax;
    else if (P[2] < fZMin) z = fZMin;
    else withinZ = true;

    if (P_in) P_in[2] = z;

    double z_dist = fabs(z-P[2]);

    // Then, we tackle x and y

    double P_tmp[2];
    double P_tnorm[2];
    double dist_tmp;

    double dist_inner = 1.e6;
    double P_inner[2] = {0,0};
    double P_inorm[2] = {0,0};

    // we find the closest inner segment to the point
    for (unsigned int i=0;i<fInnerSegments.size();i++)
    {
      dist_tmp = fInnerSegments.at(i)->DistanceTo(P,P_tmp,P_tnorm);
      if (dist_tmp<dist_inner)
      {
	dist_inner = dist_tmp;
	P_inner[0] = P_tmp[0];
	P_inner[1] = P_tmp[1];
	P_inorm[0] = P_tnorm[0];
	P_inorm[1] = P_tnorm[1];
      }
    }

    double dist_outer = 1.e6;
    double P_outer[2] = {0,0};
    double P_onorm[2] = {0,0};

    // we find the closest outer segment to the point
    for (unsigned int i=0;i<fOuterSegments.size();i++)
    {
      dist_tmp = fOuterSegments.at(i)->DistanceTo(P,P_tmp,P_tnorm);
      if (dist_tmp<dist_outer)
      {
	dist_outer = dist_tmp;
	P_outer[0] = P_tmp[0];
	P_outer[1] = P_tmp[1];
	P_onorm[0] = P_tnorm[0];
	P_onorm[1] = P_tnorm[1];
      }
    }

    // assuming both the inner and outer forms wind about the origin (or at
    // least partially do so), we determine the radial and angular values for
    // our point, and the nearest inner and outer points.
    double rad_P = sqrt(P[0]*P[0]+P[1]*P[1]);
    double rad_inner = sqrt(P_inner[0]*P_inner[0] + P_inner[1]*P_inner[1]);
    double rad_outer = sqrt(P_outer[0]*P_outer[0] + P_outer[1]*P_outer[1]);

    double total_dist_inner = sqrt(dist_inner*dist_inner + z_dist*z_dist);
    double total_dist_outer = sqrt(dist_outer*dist_outer + z_dist*z_dist);

    double theta_P = Theta(P[0],P[1]);
    double theta_min = Theta(fInnerSegments.front()->GetP1(0),
			     fInnerSegments.front()->GetP1(1));
    double theta_max = Theta(fInnerSegments.back()->GetP2(0),
			     fInnerSegments.back()->GetP2(1));

    double theta_mid = Theta(fInnerSegments.at(fInnerSegments.size()/2)->
			     GetP2(0),
			     fInnerSegments.at(fInnerSegments.size()/2)->
			     GetP2(1));

    bool orientation = true;
    {
      double thetas[2] = {fmod(theta_mid-theta_min + 2.*M_PI,2.*M_PI),
			  fmod(theta_max-theta_min + 2.*M_PI,2.*M_PI)};
      orientation = (thetas[0] < thetas[1]);
    }

    // we need to determine whether we are on one of the faces, or on the
    // extruded component of the 

    if ((rad_inner <= rad_P && rad_outer >= rad_P) &&
	Arc::AngleIsWithinRange(theta_P,theta_min,theta_max,orientation))
    {
      // here, we may be somewhere in the middle of one of the faces

      double dist_extrusion = (fabs(P[2]-fZMin) < fabs(P[2]-fZMax) ?
			       fabs(P[2]-fZMin) : fabs(P[2]-fZMax));

      if (withinZ)
      {
	// it is ambiguous, so we test the values and return the minimum

	if (total_dist_inner < total_dist_outer &&
	    total_dist_inner < dist_extrusion)
	{
	  // we are closest to the extrusion, on its inner surface
	  if (P_in)
	  {
	    P_in[0] = P_inner[0];
	    P_in[1] = P_inner[1];
	  }
	  if (P_norm)
	  {
	    P_norm[0] = P_inorm[0];
	    P_norm[1] = P_inorm[1];
	    P_norm[2] = 0.;
	  }
	  return total_dist_inner;
	}

	if (total_dist_outer < total_dist_inner &&
	    total_dist_outer < dist_extrusion)
	{
	  // we are closest to the extrusion, on its outer surface
	  if (P_in)
	  {
	    P_in[0] = P_outer[0];
	    P_in[1] = P_outer[1];
	  }
	  if (P_norm)
	  {
	    P_norm[0] = P_onorm[0];
	    P_norm[1] = P_onorm[1];
	    P_norm[2] = 0.;
	  }
	  return total_dist_outer;
	}
      }

      // we are closest to one of the faces
      if (P_in)
      {
	P_in[0] = P[0];
	P_in[1] = P[1];
	P_in[2] = (fabs(P[2]-fZMin) < fabs(P[2]-fZMax) ? fZMin : fZMax);
      }
      if (P_norm)
      {
	P_norm[0] = P_norm[1] = 0.;
	P_norm[2] = (fabs(P[2]-fZMin) < fabs(P[2]-fZMax) ? -1. : 1.);
      }
      return dist_extrusion;
    }

    // if we get here, we're definitely on the extrusion (that could mean that
    // we're on the edge of one of the faces, but it's all the same!)
    double dist = 0.;
    if (dist_inner < dist_outer)
    {
      if (P_in)
      {
	P_in[0] = P_inner[0];
	P_in[1] = P_inner[1];
      }

      dist = sqrt(dist_inner*dist_inner + z_dist*z_dist);

      if (P_norm)
      {
	P_norm[0] = P_inorm[0]*dist_inner/dist;
	P_norm[1] = P_inorm[1]*dist_inner/dist;
	P_norm[2] = z_dist/dist;
      }
    }
    else
    {
      if (P_in)
      {
	P_in[0] = P_outer[0];
	P_in[1] = P_outer[1];
      }

      dist = sqrt(dist_outer*dist_outer + z_dist*z_dist);

      if (P_norm)
      {
	P_norm[0] = P_onorm[0]*dist_inner/dist;
	P_norm[1] = P_onorm[1]*dist_inner/dist;
	P_norm[2] = z_dist/dist;
      }
    }

    return dist;
  }
    
  KGExtrudedObject::Line::Line(KGExtrudedObject* eO,
			       double p1[2],
			       double p2[2])
  {
    fExtruded = eO;

    for (int i=0;i<2;i++)
    {
      fP1[i] = p1[i];
      fP2[i] = p2[i];
    }

    Initialize();
  }

  void KGExtrudedObject::Line::Initialize() const
  {
    fLength = sqrt((fP2[0]-fP1[0])*(fP2[0]-fP1[0]) +
		   (fP2[1]-fP1[1])*(fP2[1]-fP1[1]));
  }

  KGExtrudedObject::Line* KGExtrudedObject::Line::Clone(KGExtrudedObject* eO) const
  {
    Line* tClone = new Line();

    tClone->fOrder = fOrder;
    tClone->fP1[0] = fP1[0];
    tClone->fP1[1] = fP1[1];
    tClone->fP2[0] = fP2[0];
    tClone->fP2[1] = fP2[1];
    tClone->fLength = fLength;
    tClone->fNDisc = fNDisc;
    tClone->fExtruded = eO;

    return tClone;
  }

  double KGExtrudedObject::Line::DistanceTo(const double* P,
					    double* P_in,
					    double* P_norm) const
  {
    // Assuming the point is cast onto a plane intersecting the z-axis, finds
    // the distance and closest point in 2-space

    double u = ((P[0]-fP1[0])*(fP2[0]-fP1[0]) +
		(P[1]-fP1[1])*(fP2[1]-fP1[1]))/(fLength*fLength);

    double unit[2] = {(fP2[0]-fP1[0])/fLength,
		      (fP2[1]-fP1[1])/fLength};

    double dist = 0.;

    if (u<=0.)
    {
      if (P_in!=NULL)
      {
	P_in[0] = fP1[0];
	P_in[1] = fP1[1];
      }

      dist = sqrt((P[0]-fP1[0])*(P[0]-fP1[0]) + (P[1]-fP1[1])*(P[1]-fP1[1]));
    }
    else if (u>=1.)
    {
      if (P_in!=NULL)
      {
	P_in[0] = fP2[0];
	P_in[1] = fP2[1];
      }
      dist = sqrt((P[0]-fP2[0])*(P[0]-fP2[0]) + (P[1]-fP2[1])*(P[1]-fP2[1]));
    }
    else
    {
      double x_int = fP1[0] + u*(fP2[0]-fP1[0]);
      double y_int = fP1[1] + u*(fP2[1]-fP1[1]);

      if (P_in!=NULL)
      {
	P_in[0] = x_int;
	P_in[1] = y_int;
      }

      dist = sqrt((P[0]-x_int)*(P[0]-x_int) + (P[1]-y_int)*(P[1]-y_int));
    }

    if (P_norm!=NULL)
    {
      P_norm[0] = P[0] - P_in[0];
      P_norm[1] = P[1] - P_in[1];

      double dot = (P_norm[0]*unit[0] + P_norm[1]*unit[1])/dist;

      P_norm[0] -= dot*unit[0];
      P_norm[1] -= dot*unit[1];

      double len = sqrt(P_norm[0]*P_norm[0] + P_norm[1]*P_norm[1]);
      P_norm[0] /= len;
      P_norm[1] /= len;
    }

    return dist;
  }

  KGExtrudedObject::Arc::Arc(KGExtrudedObject* eO,
			     double p1[2],
			     double p2[2],
			     double radius,
			     bool   positiveOrientation)
    : KGExtrudedObject::Line(eO,p1,p2)
  {
    fRadius = radius;
    fPositiveOrientation = positiveOrientation;

    Initialize();
  }

  void KGExtrudedObject::Arc::Initialize() const
  {
    KGExtrudedObject::Line::Initialize();

    ComputeAngles();
  }

  KGExtrudedObject::Arc* KGExtrudedObject::Arc::Clone(KGExtrudedObject* eO) const
  {
    Arc* tClone = new Arc();

    tClone->fOrder = fOrder;
    tClone->fP1[0] = fP1[0];
    tClone->fP1[1] = fP1[1];
    tClone->fP2[0] = fP2[0];
    tClone->fP2[1] = fP2[1];
    tClone->fLength = fLength;
    tClone->fNDisc = fNDisc;

    tClone->fRadius = fRadius;
    tClone->fCenter[0] = fCenter[0];
    tClone->fCenter[1] = fCenter[1];
    tClone->fPhiStart = fPhiStart;
    tClone->fPhiEnd = fPhiEnd;
    tClone->fPhiBoundary = fPhiBoundary;
    tClone->fPositiveOrientation = fPositiveOrientation;

    tClone->fExtruded = eO;

    return tClone;
  }

  double KGExtrudedObject::Arc::GetLength() const
  {
    // Returns the length of the arc.

    double chord = sqrt((fP2[0]-fP1[0])*(fP2[0]-fP1[0]) +
			(fP2[1]-fP1[1])*(fP2[1]-fP1[1]));

    double theta = 2.*asin(chord/(2.*fRadius));

    return fRadius*theta;
  }

  void KGExtrudedObject::Arc::FindCenter() const
  {
    // Finds the center of the circle from which the arc is formed

    // midpoint between p1 and p2
    double pmid[2] = {(fP1[0]+fP2[0])*.5,(fP1[1]+fP2[1])*.5};

    // unit vector pointing from p1 to p2
    double unit[2] = {fP2[0]-fP1[0],fP2[1]-fP1[1]};
    double chord = sqrt(unit[0]*unit[0] + unit[1]*unit[1]);
    for (int i=0;i<2;i++) unit[i]/=chord;

    // unit vector normal to line connecting p1 and p2
    double norm[2] = {-unit[1],unit[0]};

    if (!fPositiveOrientation)
      for (int i=0;i<2;i++) norm[i]*=-1.;

    double theta = 2.*asin(chord/(2.*fRadius));

    for (int i=0;i<2;i++)
      fCenter[i] = pmid[i] + fRadius*norm[i]*cos(theta*.5);
  }

  void KGExtrudedObject::Arc::ComputeAngles() const
  {
    FindCenter();

    fPhiStart = 0;

    if (fabs(fabs(fP1[0]-fCenter[0])-fRadius) < 1.e-6)
    {
      if (fP1[0]>fCenter[0])
	fPhiStart = 0.;
      else
	fPhiStart = M_PI;
    }
    else
      fPhiStart = acos((fP1[0]-fCenter[0])/fRadius);

    if (fP1[1]<fCenter[1])
      fPhiStart = 2.*M_PI-fPhiStart;

    fPhiEnd = 0;

    if (fabs(fabs(fP2[0]-fCenter[0])-fRadius) < 1.e-6)
    {
      if (fP2[0]>fCenter[0])
	fPhiEnd = 0.;
      else
	fPhiEnd = M_PI;
    }
    else
      fPhiEnd = acos((fP2[0]-fCenter[0])/fRadius);

    if (fP2[1]<fCenter[1])
      fPhiEnd = 2.*M_PI-fPhiEnd;

    if (fPositiveOrientation && fPhiStart>fPhiEnd)
      fPhiEnd += 2.*M_PI;

    if (!fPositiveOrientation && fPhiStart<fPhiEnd)
      fPhiStart += 2.*M_PI;

    fPhiBoundary = (fPhiStart+fPhiEnd)*.5 + M_PI;
    if (fabs(fabs(fPhiStart+fPhiEnd)-2.*M_PI)<1.e-6) fPhiBoundary = M_PI;
  }

  double KGExtrudedObject::Arc::GetAngularSpread() const
  {
    double angle = NormalizeAngle(fPhiEnd - fPhiStart);

    if (angle>M_PI)
      angle = 2.*M_PI - angle;

    return angle;
  }

  double KGExtrudedObject::Arc::DistanceTo(const double* P,
					   double* P_in,
					   double* P_norm) const
  {
    // Assuming the point is cast onto a plane intersecting the z-axis, finds
    // the distance and closest point in 2-space

    double phi_P = KGExtrudedObject::Theta((P[0]-fCenter[0]),
					   (P[1]-fCenter[1]));

    double dist = 0.;

    double unit[2];

    if (AngleIsWithinRange(phi_P,fPhiStart,fPhiEnd,fPositiveOrientation))
    {
      double tmp[2] = {fRadius*cos(phi_P) + fCenter[0],
		       fRadius*sin(phi_P) + fCenter[1]};
      if (P_in != NULL)
      {
	P_in[0] = tmp[0];
	P_in[1] = tmp[1];
      }

      if (P_norm)
      {
	unit[0] = -sin(phi_P);
	unit[1] = cos(phi_P);
      }

      dist = sqrt((P[0]-tmp[0])*(P[0]-tmp[0]) + (P[1]-tmp[1])*(P[1]-tmp[1]));
    }
    else if (AngleIsWithinRange(phi_P,fPhiEnd,fPhiBoundary,fPositiveOrientation))
    {
      if (P_in != NULL)
      {
	P_in[0] = fP2[0];
	P_in[1] = fP2[1];
      }

      if (P_norm)
      {
	unit[0] = -sin(fPhiEnd);
	unit[1] = cos(fPhiEnd);
      }

      dist = sqrt((P[0]-fP2[0])*(P[0]-fP2[0]) + (P[1]-fP2[1])*(P[1]-fP2[1]));
    }
    else
    {
      if (P_in != NULL)
      {
	P_in[0] = fP1[0];
	P_in[1] = fP1[1];
      }

      if (P_norm)
      {
	unit[0] = -sin(fPhiStart);
	unit[1] = cos(fPhiStart);
      }

      dist = sqrt((P[0]-fP1[0])*(P[0]-fP1[0]) + (P[1]-fP1[1])*(P[1]-fP1[1]));
    }

    if (P_norm!=NULL)
    {
      P_norm[0] = P[0] - P_in[0];
      P_norm[1] = P[1] - P_in[1];

      double dot = (P_norm[0]*unit[0] + P_norm[1]*unit[1])/dist;

      P_norm[0] -= dot*unit[0];
      P_norm[1] -= dot*unit[1];

      double len = sqrt(P_norm[0]*P_norm[0] + P_norm[1]*P_norm[1]);
      P_norm[0] /= len;
      P_norm[1] /= len;
    }

    return dist;
  }

  double KGExtrudedObject::Arc::NormalizeAngle(double angle) const
  {
    double normalized_angle = angle;
    while (normalized_angle>2.*M_PI) normalized_angle -= 2.*M_PI;
    while (normalized_angle < 0) normalized_angle += 2.*M_PI;
    return normalized_angle;
  }

  bool KGExtrudedObject::Arc::AngleIsWithinRange(double phi_test,
						 double phi_min,
						 double phi_max,
						 bool positiveOrientation)
  {
    // determines whether or not <phi_test> is sandwiched by <phi_min> and
    // <phi_max>.

    bool result;

    if (phi_min < phi_max)
      result = (phi_min < phi_test && phi_test < phi_max);
    else
      result = (phi_test > phi_min || phi_test < phi_max);

    if (!positiveOrientation) result = !result;
    return result;
  }

}
