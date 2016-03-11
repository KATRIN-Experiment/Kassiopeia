#include "KSIntSurfaceSpecularBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSIntSurfaceSpecularBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntSurfaceSpecularStructure =
            KSIntSurfaceSpecularBuilder::Attribute< string >( "name" ) +
            KSIntSurfaceSpecularBuilder::Attribute< double >( "reflection_loss" ) +
            KSIntSurfaceSpecularBuilder::Attribute< double >( "transmission_loss" ) +
            KSIntSurfaceSpecularBuilder::Attribute< double >( "reflection_loss_fraction" ) +
            KSIntSurfaceSpecularBuilder::Attribute< double >( "transmission_loss_fraction" ) +
            KSIntSurfaceSpecularBuilder::Attribute< double >( "probability" );

    STATICINT sKSIntSurfaceSpecularElement =
            KSRootBuilder::ComplexElement< KSIntSurfaceSpecular >( "ksint_surface_specular" );
}
