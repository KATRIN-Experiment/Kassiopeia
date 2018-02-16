#include "KSIntSurfaceUCNBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSIntSurfaceUCNBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntSurfaceUCNStructure =
            KSIntSurfaceUCNBuilder::Attribute< string >( "name" ) +
            KSIntSurfaceUCNBuilder::Attribute< double >( "spin_flip_probability" )  +
            KSIntSurfaceUCNBuilder::Attribute< double >( "transmission_probability" );

    STATICINT sKSIntSurfaceUCNElement =
            KSRootBuilder::ComplexElement< KSIntSurfaceUCN >( "ksint_surface_UCN" );
}
