/*
 * KMagnetostaticFieldMapBuilder.cc
 *
 *  Created on: 27 May 2016
 *      Author: wolfgang
 */

#include "KMagnetostaticFieldmapBuilder.hh"
#include "KMagnetostaticConstantFieldBuilder.hh"
#include "KMagneticDipoleFieldBuilder.hh"
#include "KMagnetostaticBoundaryFieldBuilder.hh"
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
        KMagnetostaticFieldmapCalculatorStructure::Attribute< string >( "name" ) +
        KMagnetostaticFieldmapCalculatorStructure::Attribute< string >( "directory" ) +
        KMagnetostaticFieldmapCalculatorStructure::Attribute< string >( "file" ) +
        KMagnetostaticFieldmapCalculatorStructure::Attribute< KEMStreamableThreeVector >( "center" ) +
        KMagnetostaticFieldmapCalculatorStructure::Attribute< KEMStreamableThreeVector >( "length" ) +
        KMagnetostaticFieldmapCalculatorStructure::Attribute< bool >( "mirror_x" ) +
        KMagnetostaticFieldmapCalculatorStructure::Attribute< bool >( "mirror_y" ) +
        KMagnetostaticFieldmapCalculatorStructure::Attribute< bool >( "mirror_z" ) +
        KMagnetostaticFieldmapCalculatorStructure::Attribute< double >( "spacing" ) +
        KMagnetostaticFieldmapCalculatorStructure::Attribute< string >( "spaces" ) +
        KMagnetostaticFieldmapCalculatorStructure::Attribute< string >( "field") +
        // support of deprecated old xml:
        KMagnetostaticFieldmapCalculatorStructure::
        ComplexElement< KMagnetoostaticConstantField >( "field_magnetic_constant" ) +
        KMagnetostaticFieldmapCalculatorStructure::
        ComplexElement< KMagneticDipoleFieldBuilder >( "field_magnetic_dipole" ) +
        KMagnetostaticFieldmapCalculatorStructure::
        ComplexElement< KMagnetostaticBoundaryFieldWithKGeoBag >( "field_electromagnet" );

STATICINT sKMagnetostaticFieldmapCalculator =
        KEMToolboxBuilder::ComplexElement< KMagnetostaticFieldmapCalculator >( "magnetic_fieldmap_calculator" );


} /* namespace katrin */
