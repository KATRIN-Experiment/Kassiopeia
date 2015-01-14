#include "KSIntSurfaceSpecularBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSIntSurfaceSpecularBuilder::~KComplexElement()
    {
    }

    static int sKSIntSurfaceSpecularStructure =
            KSIntSurfaceSpecularBuilder::Attribute< string >( "name" ) +
            KSIntSurfaceSpecularBuilder::Attribute< double >( "reflection_loss" ) +
            KSIntSurfaceSpecularBuilder::Attribute< double >( "transmission_loss" ) +
            KSIntSurfaceSpecularBuilder::Attribute< double >( "probability" );

    static int sKSIntSurfaceSpecularElement =
            KSRootBuilder::ComplexElement< KSIntSurfaceSpecular >( "ksint_surface_specular" );
}
