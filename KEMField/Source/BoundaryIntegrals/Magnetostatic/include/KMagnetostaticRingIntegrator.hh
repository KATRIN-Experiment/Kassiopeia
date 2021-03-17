#ifndef KMAGNETOSTATICRINGINTEGRATOR_DEF
#define KMAGNETOSTATICRINGINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KSurface.hh"

namespace KEMField
{
class KMagnetostaticRingIntegrator
{
  public:
    using Shape = KRing;
    using ValueType = KMagnetostaticBasis::ValueType;

    friend class KMagnetostaticConicSectionIntegrator;

    KMagnetostaticRingIntegrator() = default;
    ~KMagnetostaticRingIntegrator() = default;

    KFieldVector VectorPotential(const KRing* source, const KPosition& P) const;

    KFieldVector MagneticField(const KRing* source, const KPosition& P) const;

    KFieldVector VectorPotential(const KSymmetryGroup<KRing>* source, const KPosition& P) const;

    KFieldVector MagneticField(const KSymmetryGroup<KRing>* source, const KPosition& P) const;
};

}  // namespace KEMField

#endif /* KMAGNETOSTATICRINGINTEGRATOR_DEF */
