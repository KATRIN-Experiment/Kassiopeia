/*
 * KOpenCLElectorstaticBoundaryIntegratorFactory.hh
 *
 *  Created on: 30.08.2016
 *      Author: gosda
 */

#ifndef KOPENCLELECTROSTATICBOUNDARYINTEGRATORFACTORY_HH_
#define KOPENCLELECTROSTATICBOUNDARYINTEGRATORFACTORY_HH_

#include <string>
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"

namespace KEMField {

class KOpenCLElectrostaticBoundaryIntegratorFactory
{
public:

	static KOpenCLElectrostaticBoundaryIntegrator
	MakeDefault(KOpenCLSurfaceContainer& container);

	static KOpenCLElectrostaticBoundaryIntegrator
	MakeAnalytic(KOpenCLSurfaceContainer& container);

	static KOpenCLElectrostaticBoundaryIntegrator
	MakeNumeric(KOpenCLSurfaceContainer& container);

	static KOpenCLElectrostaticBoundaryIntegrator
	MakeRWG(KOpenCLSurfaceContainer& container);

	static KOpenCLElectrostaticBoundaryIntegrator
	Make(const std::string& name,KOpenCLSurfaceContainer& container);

	static KoclEBIConfig MakeDefaultConfig();
	static KoclEBIConfig MakeAnalyticConfig();
	static KoclEBIConfig MakeNumericConfig();
	static KoclEBIConfig MakeRWGConfig();
	static KoclEBIConfig MakeConfig(const std::string& name);
};

using KoclEBIFactory = KOpenCLElectrostaticBoundaryIntegratorFactory;

} /* namespace KEMField */

#endif /* KOPENCLELECTROSTATICBOUNDARYINTEGRATORFACTORY_HH_ */
