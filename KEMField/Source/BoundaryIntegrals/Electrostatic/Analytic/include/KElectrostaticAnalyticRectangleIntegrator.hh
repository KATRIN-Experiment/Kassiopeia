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
    using Shape = KRectangle;
    using ValueType = KElectrostaticBasis::ValueType;

    KElectrostaticAnalyticRectangleIntegrator() = default;
    ~KElectrostaticAnalyticRectangleIntegrator() override = default;

    double Potential(const KRectangle* source, const KPosition& P) const override;
    KFieldVector ElectricField(const KRectangle* source, const KPosition& P) const override;
    using KElectrostaticElementIntegrator<KRectangle>::Potential;
    using KElectrostaticElementIntegrator<KRectangle>::ElectricField;

  private:
    static double Integral_ln(double x, double y, double w);

    static double EFieldLocalXY(double x1, double x2, double y1, double y2, double z);
    static double EFieldLocalZ(double x1, double x2, double y1, double y2, double z);
};

}  // namespace KEMField

#endif /* KELECTROSTATICRECTANGLEINTEGRATOR_DEF */
