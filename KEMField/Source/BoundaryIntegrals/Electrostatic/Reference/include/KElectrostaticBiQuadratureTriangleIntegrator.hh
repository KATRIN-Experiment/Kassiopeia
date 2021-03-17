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
    using Shape = KTriangle;
    using ValueType = KElectrostaticBasis::ValueType;

    KElectrostaticBiQuadratureTriangleIntegrator() = default;
    ~KElectrostaticBiQuadratureTriangleIntegrator() override = default;

    double Potential(const KTriangle* source, const KPosition& P) const override;
    KFieldVector ElectricField(const KTriangle* source, const KPosition& P) const override;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KTriangle* source,
                                                              const KPosition& P) const override;

    double Potential(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const override;
    KFieldVector ElectricField(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const override;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KSymmetryGroup<KTriangle>* source,
                                                              const KPosition& P) const override;

  private:
    static double triQuadGaussLegendreVarN(double (*f)(double), double a, double b, unsigned int n);
    static double triF1(double y);
    static double triF2(double x);
    static double triF(const KFieldVector& Q, const KFieldVector& P);
};

}  // namespace KEMField

#endif /* KELECTROSTATICBIQUADRATURETRIANGLEINTEGRATOR_DEF */
