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

namespace katrin {

template< >
    KInducedAzimuthalElectricFieldBuilder::~KComplexElement()
    {
    }

    STATICINT sKInducedAzimuthalElectricFieldStructure =
        KInducedAzimuthalElectricFieldBuilder::Attribute< string >( "name" ) +
        KInducedAzimuthalElectricFieldBuilder::Attribute< string >( "root_field" );

    STATICINT sKInducedAzimuthalElectricField =
        KEMToolboxBuilder::ComplexElement< KInducedAzimuthalElectricField >( "induced_azimuthal_electric_field" );

} /* namespace katrin */
