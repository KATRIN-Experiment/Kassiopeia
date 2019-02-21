/*
 * KElectrostaticBoundaryIntegratorFactory.hh
 *
 *  Created on: 26 Aug 2016
 *      Author: wolfgang
 */

#ifndef KELECTROSTATICBOUNDARYINTEGRATORFACTORY_HH_
#define KELECTROSTATICBOUNDARYINTEGRATORFACTORY_HH_

#include "KElectrostaticBoundaryIntegrator.hh"
#include <string>

namespace KEMField {

class KElectrostaticBoundaryIntegratorFactory {
public:

    static KElectrostaticBoundaryIntegrator
    MakeDefault();

    // FFTM might use different default because only
    // close by evaluations are normally used
    static KElectrostaticBoundaryIntegrator
	MakeDefaultForFFTM();

    static KElectrostaticBoundaryIntegrator
    MakeAnalytic();

    static KElectrostaticBoundaryIntegrator
    MakeNumeric();

    static KElectrostaticBoundaryIntegrator
    MakeRWG();

    static KElectrostaticBoundaryIntegrator
    MakeReference();

    static KElectrostaticBoundaryIntegrator
    Make(const std::string& name);
};

using KEBIFactory = KElectrostaticBoundaryIntegratorFactory;

} /* namespace KEMField */

#endif /* KELECTROSTATICBOUNDARYINTEGRATORFACTORY_HH_ */
