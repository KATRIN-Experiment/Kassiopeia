/*
 * ExplicitSuperpositionSolutionComponent.hh
 *
 *  Created on: 27 Jun 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KEXPLICITSUPERPOSITIONSOLUTIONCOMPONENT_HH_
#define KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KEXPLICITSUPERPOSITIONSOLUTIONCOMPONENT_HH_

#include <string>

namespace KEMField {

class KExplicitSuperpositionSolutionComponent {
public:
    KExplicitSuperpositionSolutionComponent(): scale(1.0) {};
    virtual ~KExplicitSuperpositionSolutionComponent(){};

    std::string name;
    double scale;
    std::string hash;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KEXPLICITSUPERPOSITIONSOLUTIONCOMPONENT_HH_ */
