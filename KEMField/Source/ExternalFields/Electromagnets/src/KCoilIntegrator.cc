#include "KCoilIntegrator.hh"

#include "KSolenoidIntegrator.hh"
#include "KEMConstants.hh"
#include "KGaussianQuadrature.hh"

namespace KEMField
{
  KThreeVector KCoilIntegrator::VectorPotential(const KCoil& coil,const KPosition& P) const
  {
    static double (*f[1])(const double*,double*) = {&KSolenoidIntegrator::A_theta};
    static KGaussianQuadrature Quad;

    KPosition p = coil.GetCoordinateSystem().ToLocal(P);

    double par[4] = {p[2],                      // z
		     sqrt(p[0]*p[0]+p[1]*p[1]), // r
		     coil.GetP0()[2],                    // z_min
		     coil.GetP1()[2]};                   // z_max

    double I=0.;

    if (p[2]<par[3] && p[2]>par[2] && par[1]>coil.GetP0()[0] && par[1]<coil.GetP1()[0])
    {
      double tmp;
      Quad(f,1,coil.GetP0()[0],par[1]-1.e-12,par,coil.GetIntegrationScale(),&tmp);
      I+=tmp;
      Quad(f,1,par[1]+1.e-12,coil.GetP1()[0],par,coil.GetIntegrationScale(),&tmp);
      I+=tmp;
    }
    else
      Quad(f,1,coil.GetP0()[0],coil.GetP1()[0],par,coil.GetIntegrationScale(),&I);

    double a_theta = KEMConstants::Mu0OverPi*coil.GetCurrentDensity()*I;

    if (fabs(par[1])<1.e-14) par[1] = 1.e-14;
    double cos = p[0]/par[1];
    double sin = p[1]/par[1];

    return coil.GetCoordinateSystem().ToGlobal(KThreeVector(-sin*a_theta,cos*a_theta,0.));
  }

  KThreeVector KCoilIntegrator::MagneticField(const KCoil& coil, const KPosition& P) const
  {
    static double (*f[2])(const double*,double*)
      = {&KSolenoidIntegrator::B_r,&KSolenoidIntegrator::B_z};
    static KGaussianQuadrature Quad;

    KPosition p = coil.GetCoordinateSystem().ToLocal(P);

    double r = sqrt(p[0]*p[0]+p[1]*p[1]);
    double z = p[2];

    // stay 1.e-8 away from coil edge
    if (fabs(z-coil.GetP0()[2])<1.e-8 &&
	r>=coil.GetP0()[0]-(coil.GetP1()[0]-coil.GetP0()[0])*1.e-8 &&
	r<=coil.GetP1()[0]+(coil.GetP1()[0]-coil.GetP0()[0])*1.e-8)
      z=coil.GetP0()[2]-1.e-8;

    if (fabs(z-coil.GetP1()[2])<1.e-8 &&
	r>=coil.GetP0()[0]-(coil.GetP1()[0]-coil.GetP0()[0])*1.e-8 &&
	r<=coil.GetP1()[0]+(coil.GetP1()[0]-coil.GetP0()[0])*1.e-8)
      z=coil.GetP1()[2]+1.e-8;

    double par[4] = {z, // z
		     r, // r
		     coil.GetP0()[2], // z_min
		     coil.GetP1()[2]}; // z_max

    double Ir;
    double Iz;

    if (p[2]<par[3] && p[2]>par[2] && par[1]>coil.GetP0()[0] && par[1]<coil.GetP1()[0])
    {
      double tmp[2];
      Quad(f,2,coil.GetP0()[0],par[1]-1.e-12,par,coil.GetIntegrationScale(),tmp);
      Ir=tmp[0];
      Iz=tmp[1];
      Quad(f,2,par[1]+1.e-12,coil.GetP1()[0],par,coil.GetIntegrationScale(),tmp);
      Ir+=tmp[0];
      Iz+=tmp[1];
    }
    else
    {
      double tmp[2];
      Quad(f,2,coil.GetP0()[0],coil.GetP1()[0],par,coil.GetIntegrationScale(),tmp);
      Ir=tmp[0];
      Iz=tmp[1];
    }

    double b_z = -KEMConstants::Mu0OverPi*coil.GetCurrentDensity()*Iz;
    double b_r = -KEMConstants::Mu0OverPi*coil.GetCurrentDensity()*Ir;

    if (fabs(par[1])<1.e-14) par[1] = 1.e-14;
    double cosine = p[0]/par[1];
    double sine   = p[1]/par[1];

    return coil.GetCoordinateSystem().ToGlobal(KThreeVector(cosine*b_r,sine*b_r,b_z));
  }
}
