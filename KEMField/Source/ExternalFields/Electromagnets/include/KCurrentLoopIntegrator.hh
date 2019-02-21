#ifndef KCURRENTLOOPINTEGRATOR_H
#define KCURRENTLOOPINTEGRATOR_H

#include "KCurrentLoop.hh"

namespace KEMField
{
  class KCurrentLoopIntegrator
  {
  public:
    KCurrentLoopIntegrator() {}
    virtual ~KCurrentLoopIntegrator() {}

    KThreeVector VectorPotential(const KCurrentLoop& currentLoop,
				 const KPosition& P) const;
    KThreeVector MagneticField(const KCurrentLoop& currentLoop,
			       const KPosition& P) const;
  };
}

#endif /* KCURRENTLOOPINTEGRATOR */
