#ifndef KELECTROSTATICBIQUADRATURETRIANGLEINTEGRATOR_DEF
#define KELECTROSTATICBIQUADRATURETRIANGLEINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KElectrostaticElementIntegrator.hh"
#include "KGaussLegendreQuadrature.hh"
#include "KSurface.hh"
#include "KSymmetryGroup.hh"

namespace KEMField
{
class KElectrostaticBiQuadratureTriangleIntegrator : public KElectrostaticElementIntegrator<KTriangle>
{
  public:
    typedef KTriangle Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticBiQuadratureTriangleIntegrator() {}
    ~KElectrostaticBiQuadratureTriangleIntegrator() override {}

    double Potential(const KTriangle* source, const KPosition& P) const override;
    KThreeVector ElectricField(const KTriangle* source, const KPosition& P) const override;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const KTriangle* source,
                                                              const KPosition& P) const override;

    double Potential(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const override;
    KThreeVector ElectricField(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const override;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const KSymmetryGroup<KTriangle>* source,
                                                              const KPosition& P) const override;

  private:
    static double triQuadGaussLegendreVarN(double (*f)(double), double a, double b, unsigned int n);
    static double triF1(double x);
    static double triF2(double y);
    static double triF(KThreeVector x, KThreeVector y);
};

}  // namespace KEMField

#endif /* KELECTROSTATICBIQUADRATURETRIANGLEINTEGRATOR_DEF */
