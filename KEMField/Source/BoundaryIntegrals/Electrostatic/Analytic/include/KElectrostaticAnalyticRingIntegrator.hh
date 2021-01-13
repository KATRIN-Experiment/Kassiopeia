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
    using Shape = KRing;
    using ValueType = KElectrostaticBasis::ValueType;

    friend class KElectrostaticAnalyticConicSectionIntegrator;

    KElectrostaticAnalyticRingIntegrator() = default;
    ~KElectrostaticAnalyticRingIntegrator() override = default;

    double Potential(const KRing* source, const KPosition& P) const override;
    KFieldVector ElectricField(const KRing* source, const KPosition& P) const override;
    using KElectrostaticElementIntegrator<KRing>::Potential;
    using KElectrostaticElementIntegrator<KRing>::ElectricField;

  private:
    static double PotentialFromChargedRing(const double* P, const double* par);
    static double EFieldRFromChargedRing(const double* P, const double* par);
    static double EFieldZFromChargedRing(const double* P, const double* par);
};

}  // namespace KEMField

#endif /* KELECTROSTATICRINGINTEGRATOR_DEF */
