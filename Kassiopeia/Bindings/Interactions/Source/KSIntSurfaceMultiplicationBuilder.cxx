#include "KSIntSurfaceMultiplicationBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSIntSurfaceMultiplicationBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntSurfaceMultiplicationStructure =
            KSIntSurfaceMultiplicationBuilder::Attribute< string >( "name" ) +
            KSIntSurfaceMultiplicationBuilder::Attribute< string >( "side" ) +
            KSIntSurfaceMultiplicationBuilder::Attribute< double >( "energy_loss_fraction" ) +
            KSIntSurfaceMultiplicationBuilder::Attribute< double >( "required_energy_per_particle_ev" );

    STATICINT sKSIntSurfaceMultiplicationElement =
            KSRootBuilder::ComplexElement< KSIntSurfaceMultiplication >( "ksint_surface_multiplication" );
}
