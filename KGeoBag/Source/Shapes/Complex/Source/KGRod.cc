#include "KGRod.hh"

#include "KGBeam.hh"

namespace KGeoBag
{

  KGRod* KGRod::Clone() const
  {
    KGRod* r = new KGRod();

    for (unsigned int i=0;i<fCoords.size();i++)
      r->fCoords.push_back(fCoords.at(i));

    r->fRadius = fRadius;
    r->fNDiscRad = fNDiscRad;
    r->fNDiscLong = fNDiscLong;

    return r;
  }

  double KGRod::GetLength() const
  {
    double length = 0.;
    for (unsigned int i=0;i<fCoords.size()-1;i++)
      length += sqrt((fCoords.at(i+1).at(0)-fCoords.at(i).at(0))*
		     (fCoords.at(i+1).at(0)-fCoords.at(i).at(0)) +
		     (fCoords.at(i+1).at(1)-fCoords.at(i).at(1))*
		     (fCoords.at(i+1).at(1)-fCoords.at(i).at(1)) +
		     (fCoords.at(i+1).at(2)-fCoords.at(i).at(2))*
		     (fCoords.at(i+1).at(2)-fCoords.at(i).at(2)));
    return length;
  }

  double KGRod::Area() const
  {
    return 2.*M_PI*GetRadius()*GetLength();
  }

  double KGRod::Volume() const
  {
    return M_PI*GetRadius()*GetRadius()*GetLength();
  }

  void KGRod::AddPoint(double p[3])
  {
    std::vector<double> tmp(3);
    for (unsigned int i=0;i<3;i++) tmp[i] = p[i];
    fCoords.push_back(tmp);
  }

  bool KGRod::ContainsPoint(const double* P) const
  {
    for (unsigned int i=0;i<fCoords.size()-1;i++)
    {
      double dist = KGBeam::DistanceToLine(P,
					   &(fCoords.at(i).at(0)),
					   &(fCoords.at(i+2).at(0)));
      if (dist<fRadius)
	return true;
    }
    return false;
  }

  void KGRod::Normalize(const double* p1,
			const double* p2,
			double* norm)
  {
    double  len = 0.;
    for (unsigned int i=0;i<3;i++)
    {
      norm[i] = p2[i] - p1[i];
      len += norm[i]*norm[i];
    }
    len = sqrt(len);
    for (unsigned int i=0;i<3;i++)
      norm[i]/=len;
  }

  void KGRod::GetNormal(const double* p1,
			const double* p2,
			const double* oldNormal,
			double* normal)
  {
    // Given a line going through p1 and p2, returns a unit vector that lies in
    // the plane normal to the line.

    // we start by constructing the unit normal vector pointing from n1 to n2
    double n[3];
    double len = 0;
    for (unsigned int i=0;i<3;i++)
    {
      n[i] = p2[i] - p1[i];
      len += n[i]*n[i];
    }
    len = sqrt(len);
    for (unsigned int i=0;i<3;i++)
      n[i] /= len;

    if (oldNormal == NULL)
    {
      // we then start with a normal vector whose sole component lies in the
      // direction of the smallest magnitude of n
      int iSmallest = 0;
      double smallest = 2.;
      for (unsigned int i=0;i<3;i++)
      {
	if (smallest>fabs(n[i]))
	{
	  smallest = fabs(n[i]);
	  iSmallest = i;
	}
      }
      normal[0] = normal[1] = normal[2] = 0;
      normal[iSmallest] = 1.;
    }
    else
    {
      for (unsigned int i=0;i<3;i++)
	normal[i] = oldNormal[i];
    }

    // we then subtract away the parts of normal that are in the direction of n
    double ndotnormal = 0.;
    for (unsigned int i=0;i<3;i++)
      ndotnormal += n[i]*normal[i];

    if (fabs(fabs(ndotnormal)-1.)<1.e-8)
    {
      double tmp = normal[0];
      normal[0] = normal[1];
      normal[1] = normal[2];
      normal[2] = tmp;
      for (unsigned int i=0;i<3;i++)
	ndotnormal += n[i]*normal[i];
    }

    len = 0.;
    for (unsigned int i=0;i<3;i++)
    {
      normal[i] -= ndotnormal*n[i];
      len += normal[i]*normal[i];
    }
    len = sqrt(len);
    for (unsigned int i=0;i<3;i++)
      normal[i]/=len;

    return;
  }

  double KGRod::DistanceTo(const double* P,double* P_in,double* P_norm) const
  {
    // Returns the shortest distance between <P> and the rod, and sets <P_in> to
    // be the closest point on the line segment (if P_in!=NULL).

    if (P_norm && !P_in)
    {
      double p_in[3];
      return DistanceTo(P,p_in,P_norm);
    }

    double dist = 1.e6;
    double P_in_i[3];

    for (unsigned int i=0;i<fCoords.size()-1;i++)
    {
      double n1[3]; // unit vector for line
      double n2[3]; // unit vector normal to the line in the direction of P
      double len1 = 0;
      double len2 = 0;

      for (unsigned int j=0;j<3;j++)
      {
	n1[j] = fCoords.at(i+1).at(j) - fCoords.at(i).at(j);
	len1 += n1[j]*n1[j];
	n2[j] = P[j] - fCoords.at(i).at(j);
	len2 += n2[j]*n2[j];
      }
      len1 = sqrt(len1);
      len2 = sqrt(len2);

      // normalize, take the dot product of the two
      double n1dotn2 = 0;
      for (unsigned int j=0;j<3;j++)
      {
	n1[j]/=len1;
	n2[j]/=len2;
	n1dotn2 += n1[j]*n2[j];
      }

      // we subtract off the components of n1 from n2, and renormalize
      len2 = 0;
      for (unsigned int j=0;j<3;j++)
      {
	n2[j] -= n1dotn2*n1[j];
	len2 += n2[j]*n2[j];
      }
      len2 = sqrt(len2);

      for (unsigned int j=0;j<3;j++)
	n2[j]/=len2;

      double P1[3]; // starting point of line
      double P2[3]; // ending point of line

      for (unsigned int j=0;j<3;j++)
      {
	P1[j] = fCoords.at(i).at(j) + n2[j]*fRadius;
	P2[j] = fCoords.at(i+1).at(j) + n2[j]*fRadius;
      }

      double dist_i = KGBeam::DistanceToLine(P,P1,P2,P_in_i);

      if (dist_i<dist)
      {
	dist = dist_i;
	if (P_in)
	{
	  for (unsigned int j=0;j<3;j++)
	    P_in[j] = P_in_i[j];
	}

	if (P_norm)
	{
	  double tmp = 0.;
	  for (unsigned int j=0;j<3;j++)
	  {
	    P_norm[j] = P[j] - P_in_i[j];
	    tmp += P_norm[j]*n1[j];
	  }
	  double tmp2 = 0.;
	  for (unsigned int j=0;j<3;j++)
	  {
	    P_norm[j] -= tmp*n1[j];
	    tmp2 += P_norm[j]*P_norm[i];
	  }
	  tmp2 = sqrt(tmp2);
	  if (dist - fRadius)
	    tmp2 *= -1.;
	  for (unsigned int j=0;j<3;j++)
	    P_norm[j]/=tmp2;
	}
      }
    }

    return dist;
  }

}
