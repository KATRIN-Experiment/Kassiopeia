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
    //! the name of the component, just for clarification
    KExplicitSuperpositionSolutionComponentBuilder::Attribute<std::string>("name") +
    //! by what factor to scale the potential (and thus charge density values)
    //! of the precomputed charge densities.
    //!
    //! For example, there is a precomputed charge density with all surfaces set to zero potential except
    //! for one particular electrode which is at 1 Volt. To describe a situation in which
    //! this electrode is at -20 Volts, this configuration is scaled by -20 and added to the
    //! superposition solver.
    KExplicitSuperpositionSolutionComponentBuilder::Attribute<double>("scale") +
    //! the hash value of the solution for this component.
    //! These hash values are machine dependent and can be found in the
    //! output of the InspectEMFile helper program, run on the .kbd files of the
    //! individual configurations in the KEMField cache directory.
    KExplicitSuperpositionSolutionComponentBuilder::Attribute<std::string>("hash");

} /* namespace katrin */
