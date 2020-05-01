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
    typedef KRectangle Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticBiQuadratureRectangleIntegrator() {}
    ~KElectrostaticBiQuadratureRectangleIntegrator() override {}

    double Potential(const KRectangle* source, const KPosition& P) const override;
    KThreeVector ElectricField(const KRectangle* source, const KPosition& P) const override;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const KRectangle* source,
                                                              const KPosition& P) const override;

    double Potential(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const override;
    KThreeVector ElectricField(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const override;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const KSymmetryGroup<KRectangle>* source,
                                                              const KPosition& P) const override;

  private:
    static double rectQuadGaussLegendreVarN(double (*f)(double), double a, double b, unsigned int n);
    static double rectF1(double x);
    static double rectF2(double y);
    static double rectF(double x, double y);
};

}  // namespace KEMField

#endif /* KELECTROSTATICBIQUADRATURERECTANGLEINTEGRATOR_DEF */
