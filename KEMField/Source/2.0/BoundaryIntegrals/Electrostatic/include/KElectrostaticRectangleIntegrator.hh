#ifndef KELECTROSTATICRECTANGLEINTEGRATOR_DEF
#define KELECTROSTATICRECTANGLEINTEGRATOR_DEF

#include "KSurface.hh"
#include "KEMConstants.hh"
#include "KSymmetryGroup.hh"

namespace KEMField
{
  class KElectrostaticRectangleIntegrator
  {
  public:
    typedef KRectangle Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticRectangleIntegrator() {}
    ~KElectrostaticRectangleIntegrator() {}

    double Potential(const KRectangle* source,
		       const KPosition& P) const;

    KEMThreeVector ElectricField(const KRectangle* source,
			       const KPosition& P) const;

    double Potential(const KSymmetryGroup<KRectangle>* source,
		       const KPosition& P) const;

    KEMThreeVector ElectricField(const KSymmetryGroup<KRectangle>* source,
			       const KPosition& P) const;

  private:
    double Integral_ln(double x,double y,double w) const;

    double EFieldLocalXY(double x1,double x2,double y1,double y2,double z) const;
    double EFieldLocalZ(double x1,double x2,double y1,double y2,double z) const;
  };

}

#endif /* KELECTROSTATICRECTANGLEINTEGRATOR_DEF */
