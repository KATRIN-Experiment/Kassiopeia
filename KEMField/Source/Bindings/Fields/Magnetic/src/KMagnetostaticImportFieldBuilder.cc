/*
* KMagneticImportFieldBuilder.cc
*
*  Created on: 10 March 2022
*      Author: wonyong
*/

#include "KMagnetostaticImportFieldBuilder.hh"
#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin {

template<> KSFieldMagneticImportBuilder::~KComplexElement() = default;

STATICINT sKSFieldMagneticImport =
    KEMToolboxBuilder::ComplexElement< KMagnetostaticImportField >( "import_magnetic_field" );

STATICINT sKSFieldMagneticImportStructure =
    KSFieldMagneticImportBuilder::Attribute< string >( "name" ) +
    KSFieldMagneticImportBuilder::Attribute< KEMStreamableThreeVector >( "XRange" ) +
    KSFieldMagneticImportBuilder::Attribute< KEMStreamableThreeVector >( "YRange" ) +
    KSFieldMagneticImportBuilder::Attribute< KEMStreamableThreeVector >( "ZRange" );


} /* namespace katrin */
