#ifndef KELECTROMAGNETINTEGRATOR_DEF
#define KELECTROMAGNETINTEGRATOR_DEF

#include "KCoilIntegrator.hh"
#include "KCurrentLoopIntegrator.hh"
#include "KLineCurrentIntegrator.hh"
#include "KSolenoidIntegrator.hh"

namespace KEMField
{
class ElectromagnetSingleThread;

class KElectromagnetIntegrator :
    public KLineCurrentIntegrator,
    public KCurrentLoopIntegrator,
    public KSolenoidIntegrator,
    public KCoilIntegrator
{
  public:
    using KCoilIntegrator::MagneticField;
    using KCoilIntegrator::VectorPotential;
    using KCurrentLoopIntegrator::MagneticField;
    using KCurrentLoopIntegrator::VectorPotential;
    using KLineCurrentIntegrator::MagneticField;
    using KLineCurrentIntegrator::VectorPotential;
    using KSolenoidIntegrator::MagneticField;
    using KSolenoidIntegrator::VectorPotential;

    // for selection of the correct field solver template and possibly elsewhere
    using Kind = ElectromagnetSingleThread;

    KElectromagnetIntegrator() = default;
    ~KElectromagnetIntegrator() override = default;
};

}  // namespace KEMField

#endif /* KELECTROMAGNETINTEGRATOR_DEF */
