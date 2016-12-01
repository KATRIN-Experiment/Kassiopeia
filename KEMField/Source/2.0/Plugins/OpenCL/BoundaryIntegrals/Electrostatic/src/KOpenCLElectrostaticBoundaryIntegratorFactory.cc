/*
 * KOpenCLElectorstaticBoundaryIntegratorFactory.cc
 *
 *  Created on: 30.08.2016
 *      Author: gosda
 */

#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KEMSimpleException.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"

namespace KEMField
{

KOpenCLElectrostaticBoundaryIntegrator
KOpenCLElectrostaticBoundaryIntegratorFactory::MakeDefault(KOpenCLSurfaceContainer& container)
{
	return KOpenCLElectrostaticBoundaryIntegrator(MakeDefaultConfig(),container);
}

KOpenCLElectrostaticBoundaryIntegrator
KOpenCLElectrostaticBoundaryIntegratorFactory::MakeAnalytic(KOpenCLSurfaceContainer& container)
{
	return KOpenCLElectrostaticBoundaryIntegrator(MakeAnalyticConfig(),container);
}

KOpenCLElectrostaticBoundaryIntegrator
KOpenCLElectrostaticBoundaryIntegratorFactory::MakeNumeric(KOpenCLSurfaceContainer& container)
{
	return KOpenCLElectrostaticBoundaryIntegrator(MakeNumericConfig(),container);
}

KOpenCLElectrostaticBoundaryIntegrator
KOpenCLElectrostaticBoundaryIntegratorFactory::MakeRWG(KOpenCLSurfaceContainer& container)
{
	return KOpenCLElectrostaticBoundaryIntegrator(MakeRWGConfig(),container);
}

KOpenCLElectrostaticBoundaryIntegrator
KOpenCLElectrostaticBoundaryIntegratorFactory::Make( const std::string& name, KOpenCLSurfaceContainer& container)
{
	return KOpenCLElectrostaticBoundaryIntegrator(MakeConfig(name),container);
}

KoclEBIConfig KOpenCLElectrostaticBoundaryIntegratorFactory::MakeDefaultConfig()
{
	return MakeNumericConfig();
}

KoclEBIConfig KOpenCLElectrostaticBoundaryIntegratorFactory::MakeAnalyticConfig()
{
	KoclEBIConfig config;
	config.fOpenCLSourceFile = "kEMField_ElectrostaticBoundaryIntegrals.cl";
	config.fOpenCLKernelFile = "kEMField_ElectrostaticBoundaryIntegrals_kernel.cl";
	config.fOpenCLFlags = "";
	config.fCPUIntegrator = KEBIFactory::MakeAnalytic();
	return config;
}

KoclEBIConfig KOpenCLElectrostaticBoundaryIntegratorFactory::MakeNumericConfig()
{
	KoclEBIConfig config;
	config.fOpenCLSourceFile = "kEMField_ElectrostaticNumericBoundaryIntegrals.cl";
	config.fOpenCLKernelFile = "kEMField_ElectrostaticNumericBoundaryIntegrals_kernel.cl";
	std::stringstream flags;
	flags << " -DKEMFIELD_OCLFASTRWG=" << KEMFIELD_OPENCL_FASTRWG;
	config.fOpenCLFlags = flags.str();
	//config.fOpenCLFlags = " -DKEMFIELD_OCLFASTRWG=" + std::string(KEMFIELD_OPENCL_FASTRWG); /* variable defined via cmake */
	config.fCPUIntegrator = KEBIFactory::MakeNumeric();
	return config;

}

KoclEBIConfig KOpenCLElectrostaticBoundaryIntegratorFactory::MakeRWGConfig()
{
	KoclEBIConfig config;
	config.fOpenCLSourceFile = "kEMField_ElectrostaticRWGBoundaryIntegrals.cl";
	config.fOpenCLKernelFile = "kEMField_ElectrostaticRWGBoundaryIntegrals_kernel.cl";
	std::stringstream flags;
	flags << " -DKEMFIELD_OCLFASTRWG=" << KEMFIELD_OPENCL_FASTRWG;
	//config.fOpenCLFlags = " -DKEMFIELD_OCLFASTRWG=" + std::string(KEMFIELD_OPENCL_FASTRWG);
	config.fOpenCLFlags = flags.str();
	config.fCPUIntegrator = KEBIFactory::MakeRWG();
	return config;
}

KoclEBIConfig KOpenCLElectrostaticBoundaryIntegratorFactory::MakeConfig(
		const std::string& name)
{
	if(name == "numeric")
			return MakeNumericConfig();
		if(name == "analytic")
			return MakeAnalyticConfig();
		if(name == "rwg")
			return MakeRWGConfig();
		if(name == "default")
			return MakeDefaultConfig();
		throw KEMSimpleException("KOpenCLElectrostaticBoundaryIntegratorFactory has no integrator with name: " + name);
}

} /* namespace KEMField */
