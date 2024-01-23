/*
* KElectricImportFieldBuilder.cc
*
*  Created on: 10 March 2022
*      Author: wonyong
*/

#include "KElectrostaticImportFieldBuilder.hh"
#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin {

template<> KSFieldElectricImportBuilder::~KComplexElement() = default;

STATICINT sKSFieldElectricImport =
    KEMToolboxBuilder::ComplexElement< KElectrostaticImportField >( "import_electric_field" );

STATICINT sKSFieldElectricImportStructure =
    KSFieldElectricImportBuilder::Attribute< string >( "name" ) +
    KSFieldElectricImportBuilder::Attribute< KEMStreamableThreeVector >( "XRange" ) +
    KSFieldElectricImportBuilder::Attribute< KEMStreamableThreeVector >( "YRange" ) +
    KSFieldElectricImportBuilder::Attribute< KEMStreamableThreeVector >( "ZRange" );


} /* namespace katrin */
