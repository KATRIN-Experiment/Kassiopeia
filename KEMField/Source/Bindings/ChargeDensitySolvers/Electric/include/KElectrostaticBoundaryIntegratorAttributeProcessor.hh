/*
 * KElectrostaticBoundaryIntegratorAttributeProcessor.hh
 *
 *  Created on: 30.08.2016
 *      Author: gosda
 */

#ifndef KELECTROSTATICBOUNDARYINTEGRATORATTRIBUTEPROCESSOR_HH_
#define KELECTROSTATICBOUNDARYINTEGRATORATTRIBUTEPROCESSOR_HH_

#include "KContainer.hh"
#include "KEMBindingsMessage.hh"
#include "KEMSimpleException.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"

namespace katrin
{

/**
 * use the string value inside the KContainer (if KContainer does not hold a
 * string, you end up in nullptr territory).
 * to construct a KElectrostaticBoundaryIntegratorPolicy and pass
 * it to the object.
 * returns true if the policy generation is successful and false with an
 * adequat warning if the policy generation failed due to an unknown string
 * value.
 *
 * XType has to have void XType::SetIntegratorPolicy(KEBIPolicy& ) for the
 * template to work.
 *
 * Use in K...Builder::AddAttribute(KContainer aContainer()
 * like this:
 * if( aContainer->GetName() == "integrator")
 * return AddElectrostaticIntegratorPolicy(fObject,aContainer);
 */
template<class XType> bool AddElectrostaticIntegratorPolicy(XType* reciever, KContainer* aContainer)
{
    std::string integratorName = aContainer->AsString();
    try {
        KEMField::KEBIPolicy policy{integratorName};
        reciever->SetIntegratorPolicy(policy);
    }
    catch (KEMField::KEMSimpleException& exception) {
        BINDINGMSG(eWarning) << "GaussianEliminationFieldSolver cannot use"
                                " unknown integrator type <"
                             << integratorName << ">.";
        return false;
    }
    return true;
}

/**
 * same as above, but only pass the standard integrator to the reciever class
 * using XType::SetDirectIntegrator(const KElectrostaticBoundaryIntegrator&)
 */
template<class XType> bool AddElectrostaticIntegrator(XType* reciever, KContainer* aContainer)
{
    std::string integratorName = aContainer->AsString();
    try {
        KEMField::KEBIPolicy policy{integratorName};
        reciever->SetDirectIntegrator(policy.CreateIntegrator());
    }
    catch (KEMField::KEMSimpleException& exception) {
        BINDINGMSG(eWarning) << "GaussianEliminationFieldSolver cannot use"
                                " unknown integrator type <"
                             << integratorName << ">.";
        return false;
    }
    return true;
}

}  // namespace katrin


#endif /* KELECTROSTATICBOUNDARYINTEGRATORATTRIBUTEPROCESSOR_HH_ */
