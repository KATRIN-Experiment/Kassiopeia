/*
 * KElectrostaticBoundaryIntegratorPolicy.hh
 *
 *  Created on: 30.08.2016
 *      Author: gosda
 */

#ifndef KELECTROSTATICBOUNDARYINTEGRATORPOLICY_HH_
#define KELECTROSTATICBOUNDARYINTEGRATORPOLICY_HH_

#include "KElectrostaticBoundaryIntegrator.hh"

#include <string>

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#endif

namespace KEMField
{

/**
 * This class mirrors the Interfaces way of having or not having the
 * OpenCL part in them depending on the compilation flag.
 * It should be not much more (some addition selection gathering might happen)
 * than the combination of both electrostatic boundary integrator factories.
 *
 * Leave the factories in the constructors to detect the errors early (unknown
 * names, etc.) where the bindings can process them and create sensible error
 * messages.
 */
class KElectrostaticBoundaryIntegratorPolicy
{
  public:
    KElectrostaticBoundaryIntegratorPolicy();
    KElectrostaticBoundaryIntegratorPolicy(std::string name);

    KElectrostaticBoundaryIntegrator CreateIntegrator();
#ifdef KEMFIELD_USE_OPENCL
    KOpenCLElectrostaticBoundaryIntegrator CreateOpenCLIntegrator(KOpenCLSurfaceContainer& container);

    KoclEBIConfig CreateOpenCLConfig();
#endif

  private:
    KElectrostaticBoundaryIntegrator fIntegratorCPU;
#ifdef KEMFIELD_USE_OPENCL
    KOpenCLElectrostaticBoundaryIntegratorConfig fOpenCLIntegratorConfig;
#endif
};

// a short alias for lazy people
using KEBIPolicy = KElectrostaticBoundaryIntegratorPolicy;

} /* namespace KEMField */

#endif /* KELECTROSTATICBOUNDARYINTEGRATORPOLICY_HH_ */
