#ifndef KLINECURRENTINTEGRATOR_H
#define KLINECURRENTINTEGRATOR_H

#include "KLineCurrent.hh"

namespace KEMField
{
  class KLineCurrentIntegrator
  {
  public:
    KLineCurrentIntegrator() {}
    virtual ~KLineCurrentIntegrator() {}

    KEMThreeVector VectorPotential(const KLineCurrent& lineCurrent,
				 const KPosition& P) const;
    KEMThreeVector MagneticField(const KLineCurrent& lineCurrent,
			       const KPosition& P) const;
  };
}

#endif /* KLINECURRENTINTEGRATOR */
