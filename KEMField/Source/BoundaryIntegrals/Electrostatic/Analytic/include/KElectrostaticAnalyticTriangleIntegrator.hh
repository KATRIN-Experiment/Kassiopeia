#ifndef KELECTROSTATICTRIANGLEINTEGRATOR_DEF
#define KELECTROSTATICTRIANGLEINTEGRATOR_DEF

#include "KSurface.hh"
#include "KEMConstants.hh"

#include "KElectrostaticElementIntegrator.hh"

#include <cmath>

namespace KEMField
{
  class KElectrostaticAnalyticTriangleIntegrator :
          public KElectrostaticElementIntegrator<KTriangle>
  {
  public:
    typedef KTriangle Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticAnalyticTriangleIntegrator() {}
    ~KElectrostaticAnalyticTriangleIntegrator() {}


    double Potential( const KTriangle* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KTriangle* source, const KPosition& P ) const;
    using KElectrostaticElementIntegrator<KTriangle>::Potential;
    using KElectrostaticElementIntegrator<KTriangle>::ElectricField;
  private:

    double Potential_noZ(double a2,double b2,double a1,double b1,double y) const;
    double F1(double a,double b,double u) const;
    double I3(double a,double b,double u1,double u2) const;
    double I3p(double a,double b,double u1,double u2) const;
    double I4(double alpha,double gamma,double q2,double prefac,double t1,double t2) const;
    double I4(double a,double b,double u1,double u2) const;
    double I4_2(double alpha,double gamma,double prefac,double t1,double t2) const;
    double I4_2(double a,double b,double u1,double u2) const;
    double I1(double a,double b,double u1,double u2) const;
    double I6(double x,double u1, double u2) const;
    double I7(double x,double u1, double u2) const;
    double I2(double x,double u1,double u2) const;
    double J2(double a,double u1,double u2) const;
    double Local_Ex(double a0,double a1,double b0,double b1,double u0,double u1) const;
    double Local_Ey(double a0,double a1,double b0,double b1,double u0,double u1) const;
    double Local_Ez(double a0,double a1,double b0,double b1,double u0,double u1) const;

  };
}

#endif /* KELECTROSTATICTRIANGLEINTEGRATOR_DEF */
