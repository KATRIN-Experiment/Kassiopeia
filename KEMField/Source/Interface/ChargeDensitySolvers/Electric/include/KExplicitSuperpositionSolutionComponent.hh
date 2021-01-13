/*
 * ExplicitSuperpositionSolutionComponent.hh
 *
 *  Created on: 27 Jun 2016
 *      Author: wolfgang
 */

#ifndef KEXPLICITSUPERPOSITIONSOLUTIONCOMPONENT_HH_
#define KEXPLICITSUPERPOSITIONSOLUTIONCOMPONENT_HH_

#include <string>

namespace KEMField
{

class KExplicitSuperpositionSolutionComponent
{
  public:
    KExplicitSuperpositionSolutionComponent() : scale(1.0){};
    virtual ~KExplicitSuperpositionSolutionComponent() = default;
    ;

    std::string name;
    double scale;
    std::string hash;
};

} /* namespace KEMField */

#endif /* KEXPLICITSUPERPOSITIONSOLUTIONCOMPONENT_HH_ */
