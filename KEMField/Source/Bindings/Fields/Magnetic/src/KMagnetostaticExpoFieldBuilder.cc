/*
* KMagnetostaticExpoFieldBuilder.cc
*
*  Created on: 8 Nov 2017
*      Author: A. Cocco
*/

#include "KMagnetostaticExpoFieldBuilder.hh"
#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin {

template<> KSFieldMagneticExpoBuilder::~KComplexElement() = default;

STATICINT sKMagnetostaticExpoField =
   KEMToolboxBuilder::ComplexElement< KMagnetostaticExpoField >( "expo_magnetic_field" );

STATICINT sKMagnetostaticExpoFieldStructure =
   KSFieldMagneticExpoBuilder::Attribute< string >( "name" ) +
   KSFieldMagneticExpoBuilder::Attribute<double>("B0") +
   KSFieldMagneticExpoBuilder::Attribute<double>("lambda");

//KSFieldMagneticExpoBuilder::Attribute< KEMStreamableThreeVector >( "Bx" );

} /* namespace katrin */
