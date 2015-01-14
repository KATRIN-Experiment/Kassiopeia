#include "KGBeam.hh"

#include "KGExtrudedObject.hh"

namespace KGeoBag
{
  KGBeam::~KGBeam()
  {
    if (f2DTransform) delete f2DTransform;
  }

  KGBeam* KGBeam::Clone() const
  {
    KGBeam* b = new KGBeam();

    b->fNDiscRad = fNDiscRad;
    b->fNDiscLong = fNDiscLong;

    for (unsigned int i=0;i<fRadialDisc.size();i++)
      b->fRadialDisc.push_back(fRadialDisc.at(i));
    for (unsigned int i=0;i<fStartCoords.size();i++)
      b->fStartCoords.push_back(fStartCoords.at(i));
    for (unsigned int i=0;i<fEndCoords.size();i++)
      b->fEndCoords.push_back(fEndCoords.at(i));
    for (unsigned int i=0;i<f2DCoords.size();i++)
      b->f2DCoords.push_back(f2DCoords.at(i));

    b->f2DTransform = new KGCoordinateTransform(*f2DTransform);

    for (unsigned int i=0;i<3;i++)
    {
      b->fUnit[i] = fUnit[i];
      b->fPlane1Norm[i] = fPlane1Norm[i];
      b->fPlane2Norm[i] = fPlane2Norm[i];
    }

    return b;
  }

  void KGBeam::Initialize() const
  {
    // first, we compute the unit vector from the start coordinates to the end
    // coordinates
    double n1[3];
    double n2[3];
    for (unsigned int i=0;i<3;i++)
    {
      n1[i] = fStartCoords.at(1).at(i) - fStartCoords.at(0).at(i);
      n2[i] = fEndCoords.at(0).at(i) - fStartCoords.at(0).at(i);
    }

    double len = 0.;

    for (unsigned int i=0;i<3;i++)
    {
      fUnit[i] = n1[(i+1)%3]*n2[(i+2)%3] - n1[(i+2)%3]*n2[(i+1)%3];
      len += fUnit[i]*fUnit[i];
    }

    for (unsigned int i=0;i<3;i++)
      fUnit[i]/=len;

    // then, we compute the unit vector normal to the start coordinate plane

    for (unsigned int i=0;i<3;i++)
    {
      n1[i] = fStartCoords.at(1).at(i) - fStartCoords.at(0).at(i);
      n2[i] = fStartCoords.at(2).at(i) - fStartCoords.at(1).at(i);
    }

    len = 0.;

    for (unsigned int i=0;i<3;i++)
    {
      fPlane1Norm[i] = n1[(i+1)%3]*n2[(i+2)%3] - n1[(i+2)%3]*n2[(i+1)%3];
      len += fPlane1Norm[i]*fPlane1Norm[i];
    }

    for (unsigned int i=0;i<3;i++)
      fPlane1Norm[i]/=len;

    // then, we compute the unit vector normal to the end coordinate plane

    for (unsigned int i=0;i<3;i++)
    {
      n1[i] = fEndCoords.at(1).at(i) - fEndCoords.at(0).at(i);
      n2[i] = fEndCoords.at(2).at(i) - fEndCoords.at(1).at(i);
    }

    len = 0.;

    for (unsigned int i=0;i<3;i++)
    {
      fPlane2Norm[i] = n1[(i+1)%3]*n2[(i+2)%3] - n1[(i+2)%3]*n2[(i+1)%3];
      len += fPlane2Norm[i]*fPlane2Norm[i];
    }

    for (unsigned int i=0;i<3;i++)
      fPlane1Norm[i]/=len;

    // next, we compute the 2-dimensional coordinates for the polygon
    // cross-section
    len = 0.;
    double x_loc[3];
    for (unsigned int i=0;i<3;i++)
    {
      x_loc[i] = fStartCoords.at(1).at(i) - fStartCoords.at(0).at(i);
      len += x_loc[i]*x_loc[i];
    }
    for (unsigned int i=0;i<3;i++)
      x_loc[i]/=len;

    double z_loc[3] = {fPlane1Norm[0],fPlane1Norm[1],fPlane1Norm[2]};

    double y_loc[3];

    for (unsigned int i=0;i<3;i++)
      y_loc[i] = -1.*(x_loc[(i+1)%3]*z_loc[(i+2)%3] -
		      x_loc[(i+2)%3]*z_loc[(i+1)%3]);

    f2DTransform = new KGCoordinateTransform(&(fStartCoords.at(0).at(0)),
					     x_loc,y_loc,z_loc);

    double tmp[3];
    for (unsigned int i=0;i<fStartCoords.size();i++)
    {
      f2DTransform->ConvertToLocalCoords(&(fStartCoords.at(i).at(0)),
					 tmp,false);
      std::vector<double> tmpVec;
      tmpVec.push_back(tmp[0]);
      tmpVec.push_back(tmp[1]);
      f2DCoords.push_back(tmpVec);
    }

    // finally, we determine the radial discretization

    SetRadialDiscretization();
  }

  void KGBeam::AddStartLine(double p1[3],double p2[3])
  {
    if (fStartCoords.size()==0)
    {
      std::vector<double> tmp(3);
      for (unsigned int i=0;i<3;i++)
	tmp[i] = p1[i];
      fStartCoords.push_back(tmp);
    }
    else
    {
      for (unsigned int i=0;i<3;i++)
	if (fabs(p1[i]-fStartCoords.back().at(i))>1.e-6)
	{
	  //        std::stringstream s;
	  //        s<<"Line segments that comprise the start of a beam must be contiguous.";
	  //        KIOManager::GetInstance()->Message("Beam","AddStartLine",s.str(),2);
	}
    }

    std::vector<double> tmp(3);
    for (unsigned int i=0;i<3;i++)
      tmp[i] = p2[i];
    fStartCoords.push_back(tmp);
  }

  void KGBeam::AddEndLine(double p1[3],double p2[3])
  {
    if (fEndCoords.size()==0)
    {
      std::vector<double> tmp(3);
      for (unsigned int i=0;i<3;i++)
	tmp[i] = p1[i];
      fEndCoords.push_back(tmp);
    }
    else
    {
      for (unsigned int i=0;i<3;i++)
	if (fabs(p1[i]-fEndCoords.back().at(i))>1.e-6)
	{
	  //        std::stringstream s;
	  //        s<<"Line segments that comprise the end of a beam must be contiguous.";
	  //        KIOManager::GetInstance()->Message("Beam","AddEndLine",s.str(),2);
	}
    }

    std::vector<double> tmp(3);
    for (unsigned int i=0;i<3;i++)
      tmp[i] = p2[i];
    fEndCoords.push_back(tmp);
  }

  void KGBeam::SetRadialDiscretization() const
  {
    // Assigns a discretization parameter to each linear element of the
    // cross-section.

    double totalLength = 0.;
    std::vector<double> lengths;

    for (unsigned int i=0;i<fStartCoords.size()-1;i++)
    {
      double len = sqrt((fStartCoords.at(i).at(0)-fStartCoords.at(i+1).at(0))*
			(fStartCoords.at(i).at(0)-fStartCoords.at(i+1).at(0)) +
			(fStartCoords.at(i).at(1)-fStartCoords.at(i+1).at(1))*
			(fStartCoords.at(i).at(1)-fStartCoords.at(i+1).at(1)) +
			(fStartCoords.at(i).at(2)-fStartCoords.at(i+1).at(2))*
			(fStartCoords.at(i).at(2)-fStartCoords.at(i+1).at(2)));
      lengths.push_back(len);
      totalLength+=len;
    }

    for (unsigned int i=0;i<lengths.size();i++)
    {
      unsigned int radDisc = (unsigned int)(lengths.at(i)*fNDiscRad/totalLength);
      if (radDisc<1) radDisc = 1;
      fRadialDisc.push_back(radDisc);
    }
  }

  void KGBeam::LinePlaneIntersection(const double p1[3],
				     const double p2[3],
				     const double p[3],
				     const double n[3],
				     double p_int[3])
  {
    // Given a line that passes through p1 and p2 and a plane passing through p
    // and normal to n, returns the point p_int of intersection.

    double u[3];
    double w[3];
    double D = 0.;
    double N = 0.;

    for (unsigned int i=0;i<3;i++)
    {
      u[i] = p2[i] - p1[i];
      w[i] = p1[i] - p[i];
      D += n[i]*u[i];
      N -= n[i]*w[i];
    }

    double sI = N/D;

    for (unsigned int i=0;i<3;i++)
      p_int[i] = p1[i] + sI*u[i];
  }

  bool KGBeam::ContainsPoint(const double* P) const
  {
    double plane1_proj[3];
    double plane2_proj[3];

    double P2[3] = {P[0] + fUnit[0],
		    P[1] + fUnit[1],
		    P[2] + fUnit[2]};
    LinePlaneIntersection(P,P2,&fStartCoords.at(0).at(0),fPlane1Norm,plane1_proj);
    LinePlaneIntersection(P,P2,&fEndCoords.at(0).at(0),fPlane2Norm,plane2_proj);

    double dot1 = ((plane1_proj[0]-P[0])*fPlane1Norm[0] +
		   (plane1_proj[1]-P[1])*fPlane1Norm[1] +
		   (plane1_proj[2]-P[2])*fPlane1Norm[2]);

    double dot2 = ((plane2_proj[0]-P[0])*fPlane1Norm[0] +
		   (plane2_proj[1]-P[1])*fPlane1Norm[1] +
		   (plane2_proj[2]-P[2])*fPlane1Norm[2]);

    if (dot1*dot2 > 0.)
      return false;

    // Check the polygons

    double tmp[3];
    f2DTransform->ConvertToLocalCoords(plane1_proj,tmp,false);
    std::vector<double> tmpVec;
    tmpVec.push_back(tmp[0]);
    tmpVec.push_back(tmp[1]);

    return KGExtrudedObject::PointIsInPolygon(tmpVec,
					      f2DCoords,
					      0,
					      f2DCoords.size());
  }

  double KGBeam::DistanceTo(const double* P,double* P_in,double* P_norm) const
  {
    // Returns the shortest distance between <P> and the beam, and sets <P_in>
    // to be the closest point on the line segment (if P_in!=NULL).
    //
    // Algorithm:
    // For each line along the beam, compute the distance and closest point.
    // Take the closest two (contiguous), and use the closest point from each as
    // the inputs for an additional line.  return the closest point between this
    // line and the point.

    double P_[2][3];
    double dists[2] = {1.e6,1.e6};

    double tmp[2][3];
    double dist_tmp[2];

    dist_tmp[0] = DistanceToLine(P,
				 &fStartCoords.at(0).at(0),
				 &fEndCoords.at(0).at(0),
				 tmp[0]);

    for (unsigned int i=1;i<fStartCoords.size();i++)
    {
      dist_tmp[i%2] = DistanceToLine(P,
				     &fStartCoords.at(i).at(0),
				     &fEndCoords.at(i).at(0),
				     tmp[i%2]);
      if (dist_tmp[0]+dist_tmp[1]<dists[0]+dists[1])
      {
	for (unsigned int j=0;j<2;j++)
	{
	  dists[j] = dist_tmp[j];
	  for (unsigned int k=0;k<3;k++) P_[j][k] = tmp[j][k];
	}

	if (P_norm)
	{
	  double n1[3];
	  double n2[3];
	  for (unsigned int j=0;j<3;j++)
	  {
	    n1[j] = P_[1][j] - P_[0][j];
	    n2[j] = fEndCoords.at(i).at(j) - fStartCoords.at(i).at(j);
	  }

	  double len = 0.;
	  for (unsigned int j=0;j<3;j++)
	  {
	    P_norm[j] = n1[(j+1)%3]*n2[(j+2)%3] - n1[(j+2)%3]*n2[(j+1)%3];
	    len += P_norm[j]*P_norm[j];
	  }
	  for (unsigned int j=0;j<3;j++)
	    P_norm[j] /= len;
	}
      }
    }

    return DistanceToLine(P,P_[0],P_[1],P_in);
  }

  double KGBeam::DistanceToLine(const double *P,
				const double *P1,
				const double*P2,
				double *P_in)
  {
    // Returns the shortest distance between <P> and a line segment with
    // endpoints <P1> and <P2>, and sets <P_in> to be the closest point on the
    // line segment (if P_in!=NULL)

    double length = sqrt((P2[0]-P1[0])*(P2[0]-P1[0]) +
			 (P2[1]-P1[1])*(P2[1]-P1[1]) +
			 (P2[2]-P1[2])*(P2[2]-P1[2]));

    double u = ((P[0]-P1[0])*(P2[0]-P1[0]) +
		(P[1]-P1[1])*(P2[1]-P1[1]) +
		(P[2]-P1[2])*(P2[2]-P1[2]))/(length*length);

    double dist;

    if (u<=0.)
    {
      if (P_in!=NULL)
      {
	P_in[0] = P1[0];
	P_in[1] = P1[1];
	P_in[2] = P1[2];
      }

      dist = sqrt((P[0]-P1[0])*(P[0]-P1[0]) +
		  (P[1]-P1[1])*(P[1]-P1[1]) +
		  (P[2]-P1[2])*(P[2]-P1[2]));
    }
    else if (u>=1.)
    {
      if (P_in!=NULL)
      {
	P_in[0] = P2[0];
	P_in[1] = P2[1];
	P_in[2] = P2[2];
      }

      dist = sqrt((P[0]-P2[0])*(P[0]-P2[0]) +
		  (P[1]-P2[1])*(P[1]-P2[1]) +
		  (P[2]-P2[2])*(P[2]-P2[2]));
    }
    else
    {
      double P_tmp[3];
      P_tmp[0] = P1[0] + u*(P2[0]-P1[0]);
      P_tmp[1] = P1[1] + u*(P2[1]-P1[1]);
      P_tmp[2] = P1[2] + u*(P2[2]-P1[2]);

      if (P_in)
      {
	for (int j=0;j<3;j++)
	  P_in[j] = P_tmp[j];
      }

      dist = sqrt((P[0]-P_tmp[0])*(P[0]-P_tmp[0]) +
		  (P[1]-P_tmp[1])*(P[1]-P_tmp[1]) +
		  (P[2]-P_tmp[2])*(P[2]-P_tmp[2]));
    }
    return dist;
  }
}
