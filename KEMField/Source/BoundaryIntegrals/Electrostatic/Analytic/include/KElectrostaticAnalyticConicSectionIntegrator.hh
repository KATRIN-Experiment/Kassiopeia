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
    using Sshape = KConicSection;
    using ValueType = KElectrostaticBasis::ValueType;

    KElectrostaticAnalyticConicSectionIntegrator() = default;
    ~KElectrostaticAnalyticConicSectionIntegrator() override = default;

    double Potential(const KConicSection* source, const KPosition& P) const override;
    KFieldVector ElectricField(const KConicSection* source, const KPosition& P) const override;
    using KElectrostaticElementIntegrator<KConicSection>::Potential;
    using KElectrostaticElementIntegrator<KConicSection>::ElectricField;
};

}  // namespace KEMField

#endif /* KELECTROSTATICCONICSECTIONINTEGRATOR_DEF */
