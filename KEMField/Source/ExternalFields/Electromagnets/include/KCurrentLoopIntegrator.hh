#ifndef KCURRENTLOOPINTEGRATOR_H
#define KCURRENTLOOPINTEGRATOR_H

#include "KCurrentLoop.hh"

namespace KEMField
{
class KCurrentLoopIntegrator
{
  public:
    KCurrentLoopIntegrator() = default;
    virtual ~KCurrentLoopIntegrator() = default;

    KFieldVector VectorPotential(const KCurrentLoop& currentLoop, const KPosition& P) const;
    KFieldVector MagneticField(const KCurrentLoop& currentLoop, const KPosition& P) const;
};
}  // namespace KEMField

#endif /* KCURRENTLOOPINTEGRATOR */
