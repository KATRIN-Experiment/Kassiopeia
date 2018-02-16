/*
 * EBIFactory.cc
 *
 *  Created on: 26 Aug 2016
 *      Author: wolfgang
 */

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KElectrostatic256NodeQuadratureLineSegmentIntegrator.hh"
#include "KElectrostaticBiQuadratureRectangleIntegrator.hh"
#include "KElectrostaticBiQuadratureTriangleIntegrator.hh"
#include "KElectrostaticCubatureRectangleIntegrator.hh"
#include "KElectrostaticCubatureTriangleIntegrator.hh"
#include "KElectrostaticQuadratureLineSegmentIntegrator.hh"
#include "KElectrostaticAnalyticTriangleIntegrator.hh"
#include "KElectrostaticAnalyticRectangleIntegrator.hh"
#include "KElectrostaticAnalyticLineSegmentIntegrator.hh"
#include "KElectrostaticAnalyticConicSectionIntegrator.hh"
#include "KElectrostaticAnalyticRingIntegrator.hh"

#include "KElectrostaticRWGTriangleIntegrator.hh"
#include "KElectrostaticRWGRectangleIntegrator.hh"

#include "KEMSimpleException.hh"


namespace KEMField {

KElectrostaticBoundaryIntegrator
KElectrostaticBoundaryIntegratorFactory::MakeDefault()
{
    return MakeNumeric();
}

KElectrostaticBoundaryIntegrator
KElectrostaticBoundaryIntegratorFactory::MakeDefaultForFFTM()
{
    return MakeNumeric();
}

KElectrostaticBoundaryIntegrator
KElectrostaticBoundaryIntegratorFactory::MakeAnalytic()
{
    return KElectrostaticBoundaryIntegrator(
            new KElectrostaticAnalyticTriangleIntegrator,
            new KElectrostaticAnalyticRectangleIntegrator,
            new KElectrostaticAnalyticLineSegmentIntegrator,
            new KElectrostaticAnalyticConicSectionIntegrator,
            new KElectrostaticAnalyticRingIntegrator);
}

KElectrostaticBoundaryIntegrator
KElectrostaticBoundaryIntegratorFactory::MakeNumeric()
{
    return KElectrostaticBoundaryIntegrator(
            new KElectrostaticCubatureTriangleIntegrator,
            new KElectrostaticCubatureRectangleIntegrator,
            new KElectrostaticQuadratureLineSegmentIntegrator,
            new KElectrostaticAnalyticConicSectionIntegrator,
            new KElectrostaticAnalyticRingIntegrator);
}

KElectrostaticBoundaryIntegrator
KElectrostaticBoundaryIntegratorFactory::MakeRWG()
{
    return KElectrostaticBoundaryIntegrator(
            new KElectrostaticRWGTriangleIntegrator,
            new KElectrostaticRWGRectangleIntegrator,
            new KElectrostaticAnalyticLineSegmentIntegrator,
            new KElectrostaticAnalyticConicSectionIntegrator,
            new KElectrostaticAnalyticRingIntegrator);

}

KElectrostaticBoundaryIntegrator
KElectrostaticBoundaryIntegratorFactory::MakeReference()
{
    return KElectrostaticBoundaryIntegrator(
            new KElectrostaticBiQuadratureTriangleIntegrator,
            new KElectrostaticBiQuadratureRectangleIntegrator,
            new KElectrostatic256NodeQuadratureLineSegmentIntegrator,
            new KElectrostaticAnalyticConicSectionIntegrator,
            new KElectrostaticAnalyticRingIntegrator);
}

KElectrostaticBoundaryIntegrator
KElectrostaticBoundaryIntegratorFactory::Make(const std::string& name)
{
    if(name == "numeric")
        return MakeNumeric();
    if(name == "analytic")
        return MakeAnalytic();
    if(name == "rwg")
        return MakeRWG();
    if(name == "reference")
        return MakeReference();
    if(name == "default")
        return MakeDefault();
    throw KEMSimpleException("KElectrostaticBoundaryIntegratorFactory has no integrator with name: " + name);
}

} /* namespace KEMField */
