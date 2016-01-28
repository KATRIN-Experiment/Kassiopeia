#include "KSTermMinDistanceBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMinDistanceBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermMinDistanceStructure =
        KSTermMinDistanceBuilder::Attribute< string >( "name" ) +
        KSTermMinDistanceBuilder::Attribute< string >( "surfaces" ) +
        KSTermMinDistanceBuilder::Attribute< double >( "min_distance" );

    STATICINT sKSTermMinDistance =
        KSRootBuilder::ComplexElement< KSTermMinDistance >( "ksterm_min_distance" );

}
