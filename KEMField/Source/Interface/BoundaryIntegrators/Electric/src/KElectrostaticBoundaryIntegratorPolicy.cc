/*
 * KElectrostaticBoundaryIntegratorPolicy.cc
 *
 *  Created on: 30.08.2016
 *      Author: gosda
 */

#include "KElectrostaticBoundaryIntegratorPolicy.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#endif

#include "KEMCoreMessage.hh"

namespace KEMField
{

KElectrostaticBoundaryIntegratorPolicy::KElectrostaticBoundaryIntegratorPolicy() :
    fIntegratorCPU(KEBIFactory::MakeDefault())
#ifdef KEMFIELD_USE_OPENCL
    ,
    fOpenCLIntegratorConfig(KOpenCLElectrostaticBoundaryIntegratorFactory::MakeDefaultConfig())
#endif
{}

KElectrostaticBoundaryIntegratorPolicy::KElectrostaticBoundaryIntegratorPolicy(const std::string& name) :
    fIntegratorCPU(KEBIFactory::Make(name))
#ifdef KEMFIELD_USE_OPENCL
    ,
    fOpenCLIntegratorConfig(KOpenCLElectrostaticBoundaryIntegratorFactory::MakeConfig(name))
#endif
{
    kem_cout() << "Using boundary integrator policy: " << name << eom;
}

KElectrostaticBoundaryIntegrator KElectrostaticBoundaryIntegratorPolicy::CreateIntegrator()
{
    return fIntegratorCPU;
}

#ifdef KEMFIELD_USE_OPENCL
KOpenCLElectrostaticBoundaryIntegrator
KElectrostaticBoundaryIntegratorPolicy::CreateOpenCLIntegrator(KOpenCLSurfaceContainer& container)
{
    return KOpenCLElectrostaticBoundaryIntegrator(fOpenCLIntegratorConfig, container);
}

KoclEBIConfig KElectrostaticBoundaryIntegratorPolicy::CreateOpenCLConfig()
{
    return fOpenCLIntegratorConfig;
}
#endif

} /* namespace KEMField */
