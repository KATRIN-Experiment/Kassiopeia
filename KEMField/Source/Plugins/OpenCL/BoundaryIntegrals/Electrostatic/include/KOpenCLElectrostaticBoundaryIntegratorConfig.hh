/*
 * KOpenCLElectrostaticBoundaryIntegratorConfig.hh
 *
 *  Created on: 30.08.2016
 *      Author: gosda
 */

#ifndef KOPENCLELECTROSTATICBOUNDARYINTEGRATORCONFIG_HH_
#define KOPENCLELECTROSTATICBOUNDARYINTEGRATORCONFIG_HH_

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

#endif /* KOPENCLELECTROSTATICBOUNDARYINTEGRATORCONFIG_HH_ */
