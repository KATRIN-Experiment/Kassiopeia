#ifndef KCOILINTEGRATOR_H
#define KCOILINTEGRATOR_H

#include "KCoil.hh"

namespace KEMField
{
class KCoilIntegrator
{
  public:
    KCoilIntegrator() {}
    virtual ~KCoilIntegrator() {}

    KThreeVector VectorPotential(const KCoil& coil, const KPosition& P) const;
    KThreeVector MagneticField(const KCoil& coil, const KPosition& P) const;
};
}  // namespace KEMField

#endif /* KCOILINTEGRATOR */
