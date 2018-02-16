#ifndef KELECTROSTATICRECTANGLEINTEGRATOR_DEF
#define KELECTROSTATICRECTANGLEINTEGRATOR_DEF

#include "KSurface.hh"
#include "KEMConstants.hh"
#include "KSymmetryGroup.hh"

#include "KElectrostaticElementIntegrator.hh"

namespace KEMField
{
  class KElectrostaticAnalyticRectangleIntegrator :
          public KElectrostaticElementIntegrator<KRectangle>
  {
  public:
    typedef KRectangle Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticAnalyticRectangleIntegrator() {}
    ~KElectrostaticAnalyticRectangleIntegrator() {}

    double Potential( const KRectangle* source, const KPosition& P ) const;
    KEMThreeVector ElectricField( const KRectangle* source, const KPosition& P ) const;
    using KElectrostaticElementIntegrator<KRectangle>::Potential;
    using KElectrostaticElementIntegrator<KRectangle>::ElectricField;

  private:
    double Integral_ln(double x,double y,double w) const;

    double EFieldLocalXY(double x1,double x2,double y1,double y2,double z) const;
    double EFieldLocalZ(double x1,double x2,double y1,double y2,double z) const;
  };

}

#endif /* KELECTROSTATICRECTANGLEINTEGRATOR_DEF */
