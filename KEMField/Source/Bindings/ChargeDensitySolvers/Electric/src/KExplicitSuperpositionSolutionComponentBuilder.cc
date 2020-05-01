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

template<> KExplicitSuperpositionSolutionComponentBuilder::~KComplexElement() {}

STATICINT sKSExplicitSuperpositionSolutionComponentStructure =
    KExplicitSuperpositionSolutionComponentBuilder::Attribute<string>("name") +
    KExplicitSuperpositionSolutionComponentBuilder::Attribute<double>("scale") +
    KExplicitSuperpositionSolutionComponentBuilder::Attribute<string>("hash");

} /* namespace katrin */
