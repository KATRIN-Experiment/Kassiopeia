#ifndef KLINECURRENTINTEGRATOR_H
#define KLINECURRENTINTEGRATOR_H

#include "KLineCurrent.hh"

namespace KEMField
{
class KLineCurrentIntegrator
{
  public:
    KLineCurrentIntegrator() = default;
    virtual ~KLineCurrentIntegrator() = default;

    KFieldVector VectorPotential(const KLineCurrent& lineCurrent, const KPosition& P) const;
    KFieldVector MagneticField(const KLineCurrent& lineCurrent, const KPosition& P) const;
};
}  // namespace KEMField

#endif /* KLINECURRENTINTEGRATOR */
