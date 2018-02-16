/*
 * KOpenCLElectrostaticBoundaryIntegratorConfig.hh
 *
 *  Created on: 30.08.2016
 *      Author: gosda
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_OPENCL_BOUNDARYINTEGRALS_ELECTROSTATIC_INCLUDE_KOPENCLELECTROSTATICBOUNDARYINTEGRATORCONFIG_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_OPENCL_BOUNDARYINTEGRALS_ELECTROSTATIC_INCLUDE_KOPENCLELECTROSTATICBOUNDARYINTEGRATORCONFIG_HH_

#include <string>
#include "KElectrostaticBoundaryIntegratorFactory.hh"

namespace KEMField {

struct KOpenCLElectrostaticBoundaryIntegratorConfig
{
	KOpenCLElectrostaticBoundaryIntegratorConfig() :
		fCPUIntegrator(KEBIFactory::MakeDefault())
	{
	}

	std::string fOpenCLSourceFile;
	std::string fOpenCLKernelFile;
	std::string fOpenCLFlags;
	KElectrostaticBoundaryIntegrator fCPUIntegrator;
};

using KoclEBIConfig = KOpenCLElectrostaticBoundaryIntegratorConfig;

} /* KEMField */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_OPENCL_BOUNDARYINTEGRALS_ELECTROSTATIC_INCLUDE_KOPENCLELECTROSTATICBOUNDARYINTEGRATORCONFIG_HH_ */
