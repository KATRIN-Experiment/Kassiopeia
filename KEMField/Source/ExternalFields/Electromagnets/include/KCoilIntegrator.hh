#ifndef KCOILINTEGRATOR_H
#define KCOILINTEGRATOR_H

#include "KCoil.hh"

namespace KEMField
{
class KCoilIntegrator
{
  public:
    KCoilIntegrator() = default;
    virtual ~KCoilIntegrator() = default;

    KFieldVector VectorPotential(const KCoil& coil, const KPosition& P) const;
    KFieldVector MagneticField(const KCoil& coil, const KPosition& P) const;
};
}  // namespace KEMField

#endif /* KCOILINTEGRATOR */
