#ifndef KMAGNETOSTATICLINESEGMENTINTEGRATOR_DEF
#define KMAGNETOSTATICLINESEGMENTINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KSurface.hh"

namespace KEMField
{
class KMagnetostaticLineSegmentIntegrator
{
  public:
    using Shape = KLineSegment;
    using ValueType = KMagnetostaticBasis::ValueType;

    KMagnetostaticLineSegmentIntegrator() = default;
    ~KMagnetostaticLineSegmentIntegrator() = default;

    KFieldVector VectorPotential(const KLineSegment* source, const KPosition& P) const;

    KFieldVector MagneticField(const KLineSegment* source, const KPosition& P) const;

    KFieldVector VectorPotential(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const;

    KFieldVector MagneticField(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const;
};

}  // namespace KEMField

#endif /* KMAGNETOSTATICLINESEGMENTINTEGRATOR_DEF */
