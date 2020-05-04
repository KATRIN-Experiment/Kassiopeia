/*
 * KElectrostaticBoundaryIntegratorOptions.hh
 *
 *  Created on: 31.08.2016
 *      Author: gosda
 */

#ifndef KELECTROSTATICBOUNDARYINTEGRATOROPTIONS_HH_
#define KELECTROSTATICBOUNDARYINTEGRATOROPTIONS_HH_

#include "KElectrostaticBoundaryIntegratorFactory.hh"

#include <map>

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KOpenCLSurfaceContainer.hh"
#endif

namespace KEMField
{

struct IntegratorOption
{
    KElectrostaticBoundaryIntegrator (*Create)();
#ifdef KEMFIELD_USE_OPENCL
    KOpenCLElectrostaticBoundaryIntegrator (*CreateOCL)(KOpenCLSurfaceContainer&);
#endif
    std::string name;
};

#ifndef KEMFIELD_USE_OPENCL
static std::map<int, IntegratorOption> integratorOptionList{{0, {&KEBIFactory::MakeAnalytic, "analytic"}},
                                                            {1, {&KEBIFactory::MakeRWG, "RWG"}},
                                                            {2, {&KEBIFactory::MakeNumeric, "numeric"}}};
#else
static std::map<int, IntegratorOption> integratorOptionList{
    {0, {&KEBIFactory::MakeAnalytic, &KoclEBIFactory::MakeAnalytic, "analytic"}},
    {1, {&KEBIFactory::MakeRWG, &KoclEBIFactory::MakeRWG, "RWG"}},
    {2, {&KEBIFactory::MakeNumeric, &KoclEBIFactory::MakeNumeric, "numeric"}}};
#endif

}  // namespace KEMField


#endif /* KELECTROSTATICBOUNDARYINTEGRATOROPTIONS_HH_ */
