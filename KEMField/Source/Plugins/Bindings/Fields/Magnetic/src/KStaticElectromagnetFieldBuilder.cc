/*
 * KStaticElectromagnetFieldBuilder.cc
 *
 *  Created on: 26 Mar 2016
 *      Author: wolfgang
 */

#include "KStaticElectromagnetFieldBuilder.hh"
#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin {

template< >
KStaticElectromagnetFieldBuilder::~KComplexElement()
{
}

STATICINT sKStaticElectromagnetFieldStructure =
    KStaticElectromagnetFieldBuilder::Attribute< string >( "name" ) +
    KStaticElectromagnetFieldBuilder::Attribute< string >( "file" ) +
    KStaticElectromagnetFieldBuilder::Attribute< string >( "directory" ) +
    KStaticElectromagnetFieldBuilder::Attribute< string >( "system" ) +
    KStaticElectromagnetFieldBuilder::Attribute< string >( "surfaces" ) +
    KStaticElectromagnetFieldBuilder::Attribute< string >( "spaces" );

STATICINT sKStaticElectromagnetField =
    KEMToolboxBuilder::ComplexElement< KStaticElectromagnetFieldWithKGeoBag >( "electromagnet_field" );
} /* namespace katrin */
