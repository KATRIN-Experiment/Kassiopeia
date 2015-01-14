#include "KLineCurrentIntegrator.hh"

#include "KEMConstants.hh"

#include "KEMCout.hh"

namespace KEMField
{
  KEMThreeVector KLineCurrentIntegrator::VectorPotential(const KLineCurrent& lineCurrent,const KPosition& P) const
  {
    KPosition p = lineCurrent.GetCoordinateSystem().ToLocal(P);

    double r0 = (lineCurrent.GetP0() - p).Magnitude();
    double r1 = (lineCurrent.GetP1() - p).Magnitude();
    double L = (lineCurrent.GetP1() - lineCurrent.GetP0()).Magnitude();
    KDirection i = (lineCurrent.GetP1() - lineCurrent.GetP0()).Unit();

    if (1.-fabs((lineCurrent.GetP0() - p).Unit().Dot(i))<1.e-8)
      return KEMThreeVector(0.,0.,0.);

    double l = (lineCurrent.GetP0() - p).Dot(i);

    double prefac =(KEMConstants::Mu0OverPi*lineCurrent.GetCurrent()*.25*
		      log((L+l+r1)/(l+r0)));
    KEMThreeVector A = i*prefac;

    return lineCurrent.GetCoordinateSystem().ToGlobal(A);
  }

  KEMThreeVector KLineCurrentIntegrator::MagneticField(const KLineCurrent& lineCurrent,const KPosition& P) const
  {
    KPosition p = lineCurrent.GetCoordinateSystem().ToLocal(P);

    KPosition r0 = p - lineCurrent.GetP0();
    KPosition r1 = p - lineCurrent.GetP1();
    KDirection i = (lineCurrent.GetP1() - lineCurrent.GetP0()).Unit();

    if (1.-fabs((lineCurrent.GetP0() - p).Unit().Dot(i))<1.e-8)
      return KEMThreeVector(0.,0.,0.);

    double l = r0.Dot(i);

    double s = sqrt(r0.MagnitudeSquared() - l*l);

    double sinTheta0 = r0.Unit().Dot(i);
    double sinTheta1 = r1.Unit().Dot(i);

    double prefac = (KEMConstants::Mu0OverPi*lineCurrent.GetCurrent()/(4.*s)*
    		       (sinTheta1-sinTheta0));
    
    KEMThreeVector BField = r0.Cross(i).Unit()*prefac;

    return lineCurrent.GetCoordinateSystem().ToGlobal(BField);
  }
}
