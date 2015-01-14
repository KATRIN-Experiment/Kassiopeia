#include "KSIntSurfaceDiffuseBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSIntSurfaceDiffuseBuilder::~KComplexElement()
    {
    }

    static int sKSIntSurfaceDiffuseStructure =
            KSIntSurfaceDiffuseBuilder::Attribute< string >( "name" ) +
            KSIntSurfaceDiffuseBuilder::Attribute< double >( "reflection_loss" ) +
            KSIntSurfaceDiffuseBuilder::Attribute< double >( "transmission_loss" ) +
            KSIntSurfaceDiffuseBuilder::Attribute< double >( "probability" );

    static int sKSIntSurfaceDiffuseElement =
            KSRootBuilder::ComplexElement< KSIntSurfaceDiffuse >( "ksint_surface_diffuse" );
}
