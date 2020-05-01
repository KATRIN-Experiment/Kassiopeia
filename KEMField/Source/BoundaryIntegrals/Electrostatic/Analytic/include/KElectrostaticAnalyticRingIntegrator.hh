#ifndef KELECTROSTATICRINGINTEGRATOR_DEF
#define KELECTROSTATICRINGINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KElectrostaticElementIntegrator.hh"
#include "KSurface.hh"

namespace KEMField
{
class KElectrostaticAnalyticRingIntegrator : public KElectrostaticElementIntegrator<KRing>
{
  public:
    typedef KRing Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    friend class KElectrostaticAnalyticConicSectionIntegrator;

    KElectrostaticAnalyticRingIntegrator() {}
    ~KElectrostaticAnalyticRingIntegrator() override {}

    double Potential(const KRing* source, const KPosition& P) const override;
    KThreeVector ElectricField(const KRing* source, const KPosition& P) const override;
    using KElectrostaticElementIntegrator<KRing>::Potential;
    using KElectrostaticElementIntegrator<KRing>::ElectricField;

  private:
    static double PotentialFromChargedRing(const double* P, double* par);
    static double EFieldRFromChargedRing(const double* P, double* par);
    static double EFieldZFromChargedRing(const double* P, double* par);
};

}  // namespace KEMField

#endif /* KELECTROSTATICRINGINTEGRATOR_DEF */
