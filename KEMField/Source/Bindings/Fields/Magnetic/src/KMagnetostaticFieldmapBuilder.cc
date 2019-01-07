/*
 * KMagnetostaticFieldmapBuilder.cc
 *
 *  Created on: 27 May 2016
 *      Author: wolfgang
 */

#include "KMagnetostaticFieldmapBuilder.hh"

#include "KMagneticDipoleFieldBuilder.hh"
#include "KMagnetostaticConstantFieldBuilder.hh"
#include "KStaticElectromagnetFieldBuilder.hh"
#include "KEMToolboxBuilder.hh"

using namespace KEMField;
namespace katrin {

template< >
KMagnetostaticFieldmapBuilder::~KComplexElement()
{
}

STATICINT sKMagnetostaticFieldmapStructure =
        KMagnetostaticFieldmapBuilder::Attribute< string >( "name" ) +
        KMagnetostaticFieldmapBuilder::Attribute< string >( "directory" ) +
        KMagnetostaticFieldmapBuilder::Attribute< string >( "file" ) +
        KMagnetostaticFieldmapBuilder::Attribute< string >( "interpolation" );

STATICINT sKMagnetostaticFieldmap =
        KEMToolboxBuilder::ComplexElement< KMagnetostaticFieldmap >( "magnetic_fieldmap" );

////////////////////////////////////////////////////////////////////

template< >
KMagnetostaticFieldmapCalculatorBuilder::~KComplexElement()
{
}

STATICINT sKMagnetostaticFieldmapCalculatorStructure =
        KMagnetostaticFieldmapCalculatorBuilder::Attribute< string >( "name" ) +
        KMagnetostaticFieldmapCalculatorBuilder::Attribute< string >( "directory" ) +
        KMagnetostaticFieldmapCalculatorBuilder::Attribute< string >( "file" ) +
        KMagnetostaticFieldmapCalculatorBuilder::Attribute< KEMStreamableThreeVector >( "center" ) +
        KMagnetostaticFieldmapCalculatorBuilder::Attribute< KEMStreamableThreeVector >( "length" ) +
        KMagnetostaticFieldmapCalculatorBuilder::Attribute< bool >( "mirror_x" ) +
        KMagnetostaticFieldmapCalculatorBuilder::Attribute< bool >( "mirror_y" ) +
        KMagnetostaticFieldmapCalculatorBuilder::Attribute< bool >( "mirror_z" ) +
        KMagnetostaticFieldmapCalculatorBuilder::Attribute< double >( "spacing" ) +
        KMagnetostaticFieldmapCalculatorBuilder::Attribute< string >( "spaces" ) +
        KMagnetostaticFieldmapCalculatorBuilder::Attribute< string >( "field") +
        // support of deprecated old xml:
        KMagnetostaticFieldmapCalculatorBuilder::
        ComplexElement< KMagnetostaticConstantField >( "field_magnetic_constant" ) +
        KMagnetostaticFieldmapCalculatorBuilder::
        ComplexElement< KMagneticDipoleFieldBuilder >( "field_magnetic_dipole" ) +
        KMagnetostaticFieldmapCalculatorBuilder::
        ComplexElement< KStaticElectromagnetField >( "field_electromagnet" );

STATICINT sKMagnetostaticFieldmapCalculator =
        KEMToolboxBuilder::ComplexElement< KMagnetostaticFieldmapCalculator >( "magnetic_fieldmap_calculator" );


} /* namespace katrin */
