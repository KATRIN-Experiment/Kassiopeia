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
            KSIntSurfaceUCNBuilder::Attribute< double >( "eta" )  +
            KSIntSurfaceUCNBuilder::Attribute< double >( "alpha" ) +
            KSIntSurfaceUCNBuilder::Attribute< double >( "real_optical_potential" ) +
            KSIntSurfaceUCNBuilder::Attribute< double >( "correlation_length" );

    STATICINT sKSIntSurfaceUCNElement =
            KSRootBuilder::ComplexElement< KSIntSurfaceUCN >( "ksint_surface_UCN" );
}
