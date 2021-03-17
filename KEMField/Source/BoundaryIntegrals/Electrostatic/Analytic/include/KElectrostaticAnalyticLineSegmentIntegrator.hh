#ifndef KELECTROSTATICLINESEGMENTINTEGRATOR_DEF
#define KELECTROSTATICLINESEGMENTINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KElectrostaticElementIntegrator.hh"
#include "KSurface.hh"

namespace KEMField
{
class KElectrostaticAnalyticLineSegmentIntegrator : public KElectrostaticElementIntegrator<KLineSegment>
{
  public:
    using Shape = KLineSegment;
    using ValueType = KElectrostaticBasis::ValueType;

    KElectrostaticAnalyticLineSegmentIntegrator() = default;
    ~KElectrostaticAnalyticLineSegmentIntegrator() override = default;

    double Potential(const KLineSegment* source, const KPosition& P) const override;
    KFieldVector ElectricField(const KLineSegment* source, const KPosition& P) const override;
    using KElectrostaticElementIntegrator<KLineSegment>::Potential;
    using KElectrostaticElementIntegrator<KLineSegment>::ElectricField;
};

}  // namespace KEMField

#endif /* KELECTROSTATICLINESEGMENTINTEGRATOR_DEF */
