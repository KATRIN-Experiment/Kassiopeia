#ifndef KELECTROSTATICRECTANGLEINTEGRATOR_DEF
#define KELECTROSTATICRECTANGLEINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KElectrostaticElementIntegrator.hh"
#include "KSurface.hh"
#include "KSymmetryGroup.hh"

namespace KEMField
{
class KElectrostaticAnalyticRectangleIntegrator : public KElectrostaticElementIntegrator<KRectangle>
{
  public:
    typedef KRectangle Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticAnalyticRectangleIntegrator() {}
    ~KElectrostaticAnalyticRectangleIntegrator() override {}

    double Potential(const KRectangle* source, const KPosition& P) const override;
    KThreeVector ElectricField(const KRectangle* source, const KPosition& P) const override;
    using KElectrostaticElementIntegrator<KRectangle>::Potential;
    using KElectrostaticElementIntegrator<KRectangle>::ElectricField;

  private:
    double Integral_ln(double x, double y, double w) const;

    double EFieldLocalXY(double x1, double x2, double y1, double y2, double z) const;
    double EFieldLocalZ(double x1, double x2, double y1, double y2, double z) const;
};

}  // namespace KEMField

#endif /* KELECTROSTATICRECTANGLEINTEGRATOR_DEF */
