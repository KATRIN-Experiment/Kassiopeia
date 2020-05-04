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

    KThreeVector VectorPotential(const KLineCurrent& lineCurrent, const KPosition& P) const;
    KThreeVector MagneticField(const KLineCurrent& lineCurrent, const KPosition& P) const;
};
}  // namespace KEMField

#endif /* KLINECURRENTINTEGRATOR */
