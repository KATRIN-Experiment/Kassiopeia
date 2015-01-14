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

    KEMThreeVector VectorPotential(const KCurrentLoop& currentLoop,
				 const KPosition& P) const;
    KEMThreeVector MagneticField(const KCurrentLoop& currentLoop,
			       const KPosition& P) const;
  };
}

#endif /* KCURRENTLOOPINTEGRATOR */
