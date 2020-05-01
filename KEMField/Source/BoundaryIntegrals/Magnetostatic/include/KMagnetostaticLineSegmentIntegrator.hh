#ifndef KMAGNETOSTATICLINESEGMENTINTEGRATOR_DEF
#define KMAGNETOSTATICLINESEGMENTINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KSurface.hh"

namespace KEMField
{
class KMagnetostaticLineSegmentIntegrator
{
  public:
    typedef KLineSegment Shape;
    typedef KMagnetostaticBasis::ValueType ValueType;

    KMagnetostaticLineSegmentIntegrator() {}
    ~KMagnetostaticLineSegmentIntegrator() {}

    KThreeVector VectorPotential(const KLineSegment* source, const KPosition& P) const;

    KThreeVector MagneticField(const KLineSegment* source, const KPosition& P) const;

    KThreeVector VectorPotential(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const;

    KThreeVector MagneticField(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const;
};

}  // namespace KEMField

#endif /* KMAGNETOSTATICLINESEGMENTINTEGRATOR_DEF */
