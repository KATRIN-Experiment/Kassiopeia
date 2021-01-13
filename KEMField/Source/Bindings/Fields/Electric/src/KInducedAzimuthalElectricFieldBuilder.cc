/*
 * KInducedAzimuthalElectricFieldBuilder.cc
 *
 *  Created on: 15 Apr 2016
 *      Author: wolfgang
 */

#include "KInducedAzimuthalElectricFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin
{

template<> KInducedAzimuthalElectricFieldBuilder::~KComplexElement() = default;

STATICINT sKInducedAzimuthalElectricFieldStructure =
    KInducedAzimuthalElectricFieldBuilder::Attribute<std::string>("name") +
    KInducedAzimuthalElectricFieldBuilder::Attribute<std::string>("root_field");

STATICINT sKInducedAzimuthalElectricField =
    KEMToolboxBuilder::ComplexElement<KInducedAzimuthalElectricField>("induced_azimuthal_electric_field");

} /* namespace katrin */
