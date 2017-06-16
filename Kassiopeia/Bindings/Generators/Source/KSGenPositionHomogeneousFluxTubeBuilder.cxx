#include "KSGenPositionHomogeneousFluxTubeBuilder.h"
#include "KSRootBuilder.h"
#include "KSGenGeneratorCompositeBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSGenPositionHomogeneousFluxTubeBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenPositionHomogeneousFluxTubeStructure =
		KSGenPositionHomogeneousFluxTubeBuilder::Attribute< string >( "name" ) +
		KSGenPositionHomogeneousFluxTubeBuilder::Attribute< double >( "flux" ) +
		KSGenPositionHomogeneousFluxTubeBuilder::Attribute< double >( "r_max" ) +
		KSGenPositionHomogeneousFluxTubeBuilder::Attribute< double >( "z_min" ) +
		KSGenPositionHomogeneousFluxTubeBuilder::Attribute< double >( "z_max" ) +
		KSGenPositionHomogeneousFluxTubeBuilder::Attribute< double >( "phi_min" ) +
		KSGenPositionHomogeneousFluxTubeBuilder::Attribute< double >( "phi_max" ) +
		KSGenPositionHomogeneousFluxTubeBuilder::Attribute< int >( "n_integration_step" ) +
		KSGenPositionHomogeneousFluxTubeBuilder::Attribute< string >( "magnetic_field_name" );

    STATICINT sToolboxKSGenPositionHomogeneousFluxTube =
        KSRootBuilder::ComplexElement< KSGenPositionHomogeneousFluxTube >( "ksgen_position_homogeneous_flux_tube" );

    STATICINT sKSGenCompositePositionHomogeneousFluxTubeStructure =
		KSGenGeneratorCompositeBuilder::ComplexElement< KSGenPositionHomogeneousFluxTube >( "position_homogeneous_flux_tube" );
}
