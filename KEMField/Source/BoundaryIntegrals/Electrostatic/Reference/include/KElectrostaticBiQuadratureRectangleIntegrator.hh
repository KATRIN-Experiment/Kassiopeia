#ifndef KELECTROSTATICBIQUADRATURERECTANGLEINTEGRATOR_DEF
#define KELECTROSTATICBIQUADRATURERECTANGLEINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KElectrostaticElementIntegrator.hh"
#include "KGaussLegendreQuadrature.hh"
#include "KSurface.hh"
#include "KSymmetryGroup.hh"

namespace KEMField
{
class KElectrostaticBiQuadratureRectangleIntegrator : public KElectrostaticElementIntegrator<KRectangle>
{
  public:
    using Shape = KRectangle;
    using ValueType = KElectrostaticBasis::ValueType;

    KElectrostaticBiQuadratureRectangleIntegrator() = default;
    ~KElectrostaticBiQuadratureRectangleIntegrator() override = default;

    double Potential(const KRectangle* source, const KPosition& P) const override;
    KFieldVector ElectricField(const KRectangle* source, const KPosition& P) const override;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KRectangle* source,
                                                              const KPosition& P) const override;

    double Potential(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const override;
    KFieldVector ElectricField(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const override;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KSymmetryGroup<KRectangle>* source,
                                                              const KPosition& P) const override;

  private:
    static double rectQuadGaussLegendreVarN(double (*f)(double), double a, double b, unsigned int n);
    static double rectF1(double x);
    static double rectF2(double y);
    static double rectF(double x, double y);
};

}  // namespace KEMField

#endif /* KELECTROSTATICBIQUADRATURERECTANGLEINTEGRATOR_DEF */
