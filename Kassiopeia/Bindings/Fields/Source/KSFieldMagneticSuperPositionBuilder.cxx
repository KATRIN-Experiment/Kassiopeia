#include "KSFieldMagneticSuperPositionBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldMagneticSuperPositionBuilder::~KComplexElement()
    {
    }


    STATICINT sKSFieldMagneticSuperPositionDataStructure =
		KSFieldMagneticSuperPositionDataBuilder::Attribute< string >( "name" ) +
		KSFieldMagneticSuperPositionDataBuilder::Attribute< double >( "enhancement" );


    STATICINT sKSFieldMagneticSuperPositionStructure =
		KSFieldMagneticSuperPositionBuilder::Attribute< string >( "name" ) +
		KSFieldMagneticSuperPositionBuilder::Attribute< bool >( "use_caching" ) +
		KSFieldMagneticSuperPositionBuilder::ComplexElement< KSFieldMagneticSuperPositionData >( "add_field" );


    STATICINT sKSFieldMagneticSuperPosition =
        KSRootBuilder::ComplexElement< KSFieldMagneticSuperPosition >( "ksfield_magnetic_super_position" );

}
