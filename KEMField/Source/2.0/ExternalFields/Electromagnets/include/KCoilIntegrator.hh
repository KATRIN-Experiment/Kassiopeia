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

    KEMThreeVector VectorPotential(const KCoil& coil, const KPosition& P) const;
    KEMThreeVector MagneticField(const KCoil& coil, const KPosition& P) const;
  };
}

#endif /* KCOILINTEGRATOR */
