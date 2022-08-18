/*
 * EBIFactory.cc
 *
 *  Created on: 26 Aug 2016
 *      Author: wolfgang
 */

#include "KElectrostaticBoundaryIntegratorFactory.hh"

#include "KEMSimpleException.hh"
#include "KElectrostatic256NodeQuadratureLineSegmentIntegrator.hh"
#include "KElectrostaticAnalyticConicSectionIntegrator.hh"
#include "KElectrostaticAnalyticLineSegmentIntegrator.hh"
#include "KElectrostaticAnalyticRectangleIntegrator.hh"
#include "KElectrostaticAnalyticRingIntegrator.hh"
#include "KElectrostaticAnalyticTriangleIntegrator.hh"
#include "KElectrostaticBiQuadratureRectangleIntegrator.hh"
#include "KElectrostaticBiQuadratureTriangleIntegrator.hh"
#include "KElectrostaticCubatureRectangleIntegrator.hh"
#include "KElectrostaticCubatureTriangleIntegrator.hh"
#include "KElectrostaticQuadratureLineSegmentIntegrator.hh"
#include "KElectrostaticRWGRectangleIntegrator.hh"
#include "KElectrostaticRWGTriangleIntegrator.hh"


namespace KEMField
{

KElectrostaticBoundaryIntegrator KElectrostaticBoundaryIntegratorFactory::MakeDefault()
{
    return MakeNumeric();
}

KElectrostaticBoundaryIntegrator KElectrostaticBoundaryIntegratorFactory::MakeDefaultForFFTM()
{
    return MakeNumeric();
}

KElectrostaticBoundaryIntegrator KElectrostaticBoundaryIntegratorFactory::MakeAnalytic()
{
    return KElectrostaticBoundaryIntegrator(std::make_shared<KElectrostaticAnalyticTriangleIntegrator>(),
                                            std::make_shared<KElectrostaticAnalyticRectangleIntegrator>(),
                                            std::make_shared<KElectrostaticAnalyticLineSegmentIntegrator>(),
                                            std::make_shared<KElectrostaticAnalyticConicSectionIntegrator>(),
                                            std::make_shared<KElectrostaticAnalyticRingIntegrator>());
}

KElectrostaticBoundaryIntegrator KElectrostaticBoundaryIntegratorFactory::MakeNumeric()
{
    return KElectrostaticBoundaryIntegrator(std::make_shared<KElectrostaticCubatureTriangleIntegrator>(),
                                            std::make_shared<KElectrostaticCubatureRectangleIntegrator>(),
                                            std::make_shared<KElectrostaticQuadratureLineSegmentIntegrator>(),
                                            std::make_shared<KElectrostaticAnalyticConicSectionIntegrator>(),
                                            std::make_shared<KElectrostaticAnalyticRingIntegrator>());
}

KElectrostaticBoundaryIntegrator KElectrostaticBoundaryIntegratorFactory::MakeRWG()
{
    return KElectrostaticBoundaryIntegrator(std::make_shared<KElectrostaticRWGTriangleIntegrator>(),
                                            std::make_shared<KElectrostaticRWGRectangleIntegrator>(),
                                            std::make_shared<KElectrostaticAnalyticLineSegmentIntegrator>(),
                                            std::make_shared<KElectrostaticAnalyticConicSectionIntegrator>(),
                                            std::make_shared<KElectrostaticAnalyticRingIntegrator>());
}

KElectrostaticBoundaryIntegrator KElectrostaticBoundaryIntegratorFactory::MakeReference()
{
    return KElectrostaticBoundaryIntegrator(std::make_shared<KElectrostaticBiQuadratureTriangleIntegrator>(),
                                            std::make_shared<KElectrostaticBiQuadratureRectangleIntegrator>(),
                                            std::make_shared<KElectrostatic256NodeQuadratureLineSegmentIntegrator>(),
                                            std::make_shared<KElectrostaticAnalyticConicSectionIntegrator>(),
                                            std::make_shared<KElectrostaticAnalyticRingIntegrator>());
}

KElectrostaticBoundaryIntegrator KElectrostaticBoundaryIntegratorFactory::Make(const std::string& name)
{
    if (name == "numeric")
        return MakeNumeric();
    if (name == "analytic")
        return MakeAnalytic();
    if (name == "rwg")
        return MakeRWG();
    if (name == "reference")
        return MakeReference();
    if (name == "default")
        return MakeDefault();
    throw KEMSimpleException("KElectrostaticBoundaryIntegratorFactory has no integrator with name: " + name);
}

} /* namespace KEMField */
