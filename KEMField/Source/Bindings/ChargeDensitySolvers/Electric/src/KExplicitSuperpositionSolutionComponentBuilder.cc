/*
 * KExplicitSuperpositionSolutionComponentBuilder.cc
 *
 *  Created on: 27 Jun 2016
 *      Author: wolfgang
 */

#include "KExplicitSuperpositionSolutionComponentBuilder.hh"

using namespace std;

namespace katrin
{

template<> KExplicitSuperpositionSolutionComponentBuilder::~KComplexElement() = default;

STATICINT sKSExplicitSuperpositionSolutionComponentStructure =
    KExplicitSuperpositionSolutionComponentBuilder::Attribute<std::string>("name") +
    KExplicitSuperpositionSolutionComponentBuilder::Attribute<double>("scale") +
    KExplicitSuperpositionSolutionComponentBuilder::Attribute<std::string>("hash");

} /* namespace katrin */
