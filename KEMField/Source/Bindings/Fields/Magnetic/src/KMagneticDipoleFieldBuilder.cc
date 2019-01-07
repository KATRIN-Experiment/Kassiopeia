/*
 * KMagneticDipoleFieldBuilder.cc
 *
 *  Created on: 24 Mar 2016
 *      Author: wolfgang
 */

#include "KMagneticDipoleFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin {

template< >
KMagneticDipoleFieldBuilder::~KComplexElement()
{
}

STATICINT sKMagneticDipoleFieldStructure =
    KMagneticDipoleFieldBuilder::Attribute< string >( "name" ) +
    KMagneticDipoleFieldBuilder::Attribute< KEMStreamableThreeVector >( "location" ) +
    KMagneticDipoleFieldBuilder::Attribute< KEMStreamableThreeVector >( "moment" );

STATICINT sKMagneticDipoleField =
        KEMToolboxBuilder::ComplexElement< KMagneticDipoleField >( "magnetic_dipole_field" );

} /* namespace katrin */
