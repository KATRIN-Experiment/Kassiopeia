#include "KGConicalWireArray.hh"

#include "KGExtrudedObject.hh"

namespace KGeoBag
{

  KGConicalWireArray* KGConicalWireArray::Clone() const
  {
    KGConicalWireArray* w = new KGConicalWireArray();

    w->fR1 = fR1;
    w->fZ1 = fZ1;
    w->fR2 = fR2;
    w->fZ2 = fZ2;
    w->fNWires = fNWires;
    w->fThetaStart = fThetaStart;
    w->fDiameter = fDiameter;
    w->fNDisc = fNDisc;
    w->fNDiscPower = fNDiscPower;

    return w;
  }

  double KGConicalWireArray::GetLength() const
  {
    return sqrt((fR2-fR1)*(fR2-fR1) + (fZ2-fZ1)*(fZ2-fZ1));
  }

  double KGConicalWireArray::Area() const
  {
    return fNWires*M_PI*fDiameter*GetLength();
  }

  double KGConicalWireArray::Volume() const
  {
    return 0.25*fNWires*M_PI*fDiameter*fDiameter*GetLength();
  }

  bool KGConicalWireArray::ContainsPoint(const double* P) const
  {
    if (DistanceTo(P) < fDiameter*.5)
      return true;
    return false;
  }

  double KGConicalWireArray::DistanceTo(const double* P,double* P_in,double* P_norm) const
  {
    // Returns the shortest distance between <P> and the rod, and sets <P_in> to
    // be the closest point on the line segment (if P_in!=NULL).

    if (!P_in && P_norm)
    {
      double p_in[3];
      return DistanceTo(P,p_in,P_norm);
    }

    double p_radius = sqrt(P[0]*P[0] + P[1]*P[1]);
    double p_theta = KGExtrudedObject::Theta(P[0],P[1]);
    double p_z = P[2];

    bool withinZ = true;
    (void) withinZ; //remove compiler warning
    double w_z = P[2];
    double z_limits[2] = {fZ1,fZ2};
    double d_theta[2] = {2.*M_PI*fR1/fNWires,2.*M_PI*fR2/fNWires};
    if (fZ1 > fZ2)
    {
      z_limits[0] = fZ2;
      z_limits[1] = fZ1;
      double tmp = d_theta[0];
      d_theta[0] = d_theta[1];
      d_theta[1] = tmp;
    }
    if (w_z < z_limits[0])
    {
      w_z = z_limits[0];
      withinZ = false;
    }
    if (w_z > z_limits[1])
    {
      w_z = z_limits[1];
      withinZ = false;
    }

    bool withinR = false;
    double ratio = ((w_z-z_limits[0])/(z_limits[1]-z_limits[0]));
    if (ratio < 0.) ratio = 0.;
    else if (ratio > 1.) ratio = 1.;
    double dTheta = (d_theta[0]*(1.-ratio) + d_theta[1]*ratio);
    double w_radius = (fR1*(1.-ratio) + fR2*ratio);
    if (p_radius>=(w_radius-fDiameter*.5) && p_radius<=(w_radius+fDiameter*.5))
      withinR = true;

    bool withinTheta = false;
    double w_theta = 0.;

    {
      double tmp = (p_theta - fThetaStart)/dTheta;

      int n = floor(tmp);
      if (tmp-n>=.5) n++;

      w_theta = fThetaStart + n*dTheta;
      if (w_theta >=2.*M_PI) w_theta -= 2.*M_PI;

      if (fabs(w_theta-p_theta)<fDiameter/w_radius*.5)
	withinTheta = true;
    }

    if (P_in)
    {
      P_in[0] = w_radius*cos(w_theta);
      P_in[1] = w_radius*sin(w_theta);
      P_in[2] = w_z;

      if (!withinTheta && !withinR)
      {
	double dir[3] = {P[0]-P_in[0],P[1]-P_in[1],0.};
	double mag = sqrt(dir[0]*dir[0]+dir[1]*dir[1]);
	P_in[0] += dir[0]/mag*fDiameter*.5;
	P_in[1] += dir[1]/mag*fDiameter*.5;
      }
    }

    if (P_norm)
    {
      P_norm[0] = P[0] - P_in[0];
      P_norm[1] = P[1] - P_in[1];
      P_norm[2] = P[2] - P_in[2];
      double unit[3] = {(fR2-fR2)*cos(w_theta),
			(fR2-fR1)*sin(w_theta),
			(fZ2-fZ1)};
      double dot = P_norm[0]*unit[0] + P_norm[1]*unit[1] + P_norm[2]*unit[2];
      double len = 0.;
      for (unsigned int i=0;i<3;i++)
      {
	P_norm[i] -= dot*unit[i];
	len += P_norm[i]*P_norm[i];
      }
      for (unsigned int i=0;i<3;i++)
	P_norm[i]/=len;
    }

    return sqrt(p_radius*p_radius +
		w_radius*w_radius +
		(p_z-w_z)*(p_z-w_z) -
		2.*p_radius*w_radius*cos(p_theta-w_theta))-fDiameter*.5;
  }

}
