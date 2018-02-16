#ifndef KELECTROMAGNETINTEGRATOR_DEF
#define KELECTROMAGNETINTEGRATOR_DEF

#include "KLineCurrentIntegrator.hh"
#include "KCurrentLoopIntegrator.hh"
#include "KSolenoidIntegrator.hh"
#include "KCoilIntegrator.hh"

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
    using KLineCurrentIntegrator::VectorPotential;
    using KCurrentLoopIntegrator::VectorPotential;
    using KSolenoidIntegrator::VectorPotential;
    using KCoilIntegrator::VectorPotential;
    using KLineCurrentIntegrator::MagneticField;
    using KCurrentLoopIntegrator::MagneticField;
    using KSolenoidIntegrator::MagneticField;
    using KCoilIntegrator::MagneticField;

    // for selection of the correct field solver template and possibly elsewhere
    typedef ElectromagnetSingleThread Kind;

    KElectromagnetIntegrator() {}
    virtual ~KElectromagnetIntegrator() {}
  };

}

#endif /* KELECTROMAGNETINTEGRATOR_DEF */
