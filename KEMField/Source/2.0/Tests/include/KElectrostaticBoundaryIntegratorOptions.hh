/*
 * KElectrostaticBoundaryIntegratorOptions.hh
 *
 *  Created on: 31.08.2016
 *      Author: gosda
 */

#ifndef KEMFIELD_SOURCE_2_0_TESTS_INCLUDE_KELECTROSTATICBOUNDARYINTEGRATOROPTIONS_HH_
#define KEMFIELD_SOURCE_2_0_TESTS_INCLUDE_KELECTROSTATICBOUNDARYINTEGRATOROPTIONS_HH_

#include <map>
#include "KElectrostaticBoundaryIntegratorFactory.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#endif

namespace KEMField {

struct IntegratorOption
{
    KElectrostaticBoundaryIntegrator (*Create)();
#ifdef KEMFIELD_USE_OPENCL
    KOpenCLElectrostaticBoundaryIntegrator (*CreateOCL)(KOpenCLSurfaceContainer&);
#endif
    std::string name;
};

#ifndef KEMFIELD_USE_OPENCL
static std::map<int,IntegratorOption> integratorOptionList {
    {0, {&KEBIFactory::MakeAnalytic,"analytic"}},
    {1, {&KEBIFactory::MakeRWG,"RWG"}},
    {2, {&KEBIFactory::MakeNumeric,"numeric"}}
};
#else
static std::map<int,IntegratorOption> integratorOptionList {
	{0, {&KEBIFactory::MakeAnalytic,&KoclEBIFactory::MakeAnalytic,"analytic"}},
	{1, {&KEBIFactory::MakeRWG,&KoclEBIFactory::MakeRWG,"RWG"}},
	{2, {&KEBIFactory::MakeNumeric,&KoclEBIFactory::MakeNumeric,"numeric"}}
};
#endif

} /* KEMField */


#endif /* KEMFIELD_SOURCE_2_0_TESTS_INCLUDE_KELECTROSTATICBOUNDARYINTEGRATOROPTIONS_HH_ */
