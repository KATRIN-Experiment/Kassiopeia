/*
 * KMagneticSuperpositionFieldBuilder.cc
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#include "KMagneticSuperpositionFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin {

STATICINT sKMagneticSuperpositionFieldEntryStructure =
        KMagneticSuperpositionFieldEntryBuilder::Attribute< string >( "name" ) +
        KMagneticSuperpositionFieldEntryBuilder::Attribute< double >( "enhancement" );


STATICINT sKMagneticSuperpositionFieldStructure =
        KMagneticSuperpositionFieldBuilder::Attribute< string >( "name" ) +
        KMagneticSuperpositionFieldBuilder::Attribute< bool >( "use_caching" ) +
        KMagneticSuperpositionFieldBuilder::ComplexElement< KMagneticSuperpositionFieldEntry >( "add_field" );


STATICINT sKEMFieldToolboxStructure =
        KEMToolboxBuilder::ComplexElement<KMagneticSuperpositionField>("magnetic_superposition_field");

} /* namespace katrin */
