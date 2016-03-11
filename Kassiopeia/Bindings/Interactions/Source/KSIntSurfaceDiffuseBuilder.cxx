#include "KSIntSurfaceDiffuseBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSIntSurfaceDiffuseBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntSurfaceDiffuseStructure =
            KSIntSurfaceDiffuseBuilder::Attribute< string >( "name" ) +
            KSIntSurfaceDiffuseBuilder::Attribute< double >( "reflection_loss" ) +
            KSIntSurfaceDiffuseBuilder::Attribute< double >( "transmission_loss" ) +
            KSIntSurfaceDiffuseBuilder::Attribute< double >( "reflection_loss_fraction" ) +
            KSIntSurfaceDiffuseBuilder::Attribute< double >( "transmission_loss_fraction" ) +
            KSIntSurfaceDiffuseBuilder::Attribute< double >( "probability" );

    STATICINT sKSIntSurfaceDiffuseElement =
            KSRootBuilder::ComplexElement< KSIntSurfaceDiffuse >( "ksint_surface_diffuse" );
}
