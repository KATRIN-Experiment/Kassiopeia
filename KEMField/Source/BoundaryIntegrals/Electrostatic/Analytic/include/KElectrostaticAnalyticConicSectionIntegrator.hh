#ifndef KELECTROSTATICCONICSECTIONINTEGRATOR_DEF
#define KELECTROSTATICCONICSECTIONINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KElectrostaticElementIntegrator.hh"
#include "KSurface.hh"

namespace KEMField
{
class KElectrostaticAnalyticConicSectionIntegrator : public KElectrostaticElementIntegrator<KConicSection>
{
  public:
    typedef KConicSection Shape;
    typedef KElectrostaticBasis::ValueType ValueType;

    KElectrostaticAnalyticConicSectionIntegrator() {}
    ~KElectrostaticAnalyticConicSectionIntegrator() override {}

    double Potential(const KConicSection* source, const KPosition& P) const override;
    KThreeVector ElectricField(const KConicSection* source, const KPosition& P) const override;
    using KElectrostaticElementIntegrator<KConicSection>::Potential;
    using KElectrostaticElementIntegrator<KConicSection>::ElectricField;
};

}  // namespace KEMField

#endif /* KELECTROSTATICCONICSECTIONINTEGRATOR_DEF */
