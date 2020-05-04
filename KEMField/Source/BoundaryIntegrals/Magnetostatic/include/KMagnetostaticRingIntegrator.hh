#ifndef KMAGNETOSTATICRINGINTEGRATOR_DEF
#define KMAGNETOSTATICRINGINTEGRATOR_DEF

#include "KEMConstants.hh"
#include "KSurface.hh"

namespace KEMField
{
class KMagnetostaticRingIntegrator
{
  public:
    typedef KRing Shape;
    typedef KMagnetostaticBasis::ValueType ValueType;

    friend class KMagnetostaticConicSectionIntegrator;

    KMagnetostaticRingIntegrator() {}
    ~KMagnetostaticRingIntegrator() {}

    KThreeVector VectorPotential(const KRing* source, const KPosition& P) const;

    KThreeVector MagneticField(const KRing* source, const KPosition& P) const;

    KThreeVector VectorPotential(const KSymmetryGroup<KRing>* source, const KPosition& P) const;

    KThreeVector MagneticField(const KSymmetryGroup<KRing>* source, const KPosition& P) const;
};

}  // namespace KEMField

#endif /* KMAGNETOSTATICRINGINTEGRATOR_DEF */
