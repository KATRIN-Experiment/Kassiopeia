#include "KSGenPositionFluxTubeBuilder.h"
#include "KSRootBuilder.h"
#include "KSGenGeneratorCompositeBuilder.h"

#include "KSGenValueFixBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSGenValueGaussBuilder.h"

#ifdef Kassiopeia_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#endif

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSGenPositionFluxTubeBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenPositionFluxTubeStructure =
		KSGenPositionFluxTubeBuilder::Attribute< string >( "name" ) +
		KSGenPositionFluxTubeBuilder::Attribute< string >( "phi" ) +
		KSGenPositionFluxTubeBuilder::Attribute< string >( "z" ) +
		KSGenPositionFluxTubeBuilder::Attribute< double >( "flux" ) +
		KSGenPositionFluxTubeBuilder::Attribute< int >( "n_integration_step" ) +
		KSGenPositionFluxTubeBuilder::Attribute< bool >( "only_surface" ) +
		KSGenPositionFluxTubeBuilder::Attribute< string >( "magnetic_field_name" ) +
		KSGenPositionFluxTubeBuilder::ComplexElement< KSGenValueFix >( "phi_fix" ) +
		KSGenPositionFluxTubeBuilder::ComplexElement< KSGenValueSet >( "phi_set" ) +
		KSGenPositionFluxTubeBuilder::ComplexElement< KSGenValueUniform >( "phi_uniform" ) +
		KSGenPositionFluxTubeBuilder::ComplexElement< KSGenValueGauss >( "phi_gauss" ) +
		KSGenPositionFluxTubeBuilder::ComplexElement< KSGenValueFix >( "z_fix" ) +
		KSGenPositionFluxTubeBuilder::ComplexElement< KSGenValueSet >( "z_set" ) +
		KSGenPositionFluxTubeBuilder::ComplexElement< KSGenValueUniform >( "z_uniform" ) +
		KSGenPositionFluxTubeBuilder::ComplexElement< KSGenValueGauss >( "z_gauss" );

#ifdef Kassiopeia_USE_ROOT
    STATICINT sKSGenPositionFluxTubeStructureROOT =
    		KSGenPositionFluxTubeBuilder::ComplexElement< KSGenValueFormula >( "r_formula" ) +
    		KSGenPositionFluxTubeBuilder::ComplexElement< KSGenValueFormula >( "z_formula" );
#endif

    STATICINT sToolboxKSGenPositionFluxTube =
        KSRootBuilder::ComplexElement< KSGenPositionFluxTube >( "ksgen_position_flux_tube" );

    STATICINT sKSGenCompositePositionFluxTubeStructure =
		KSGenGeneratorCompositeBuilder::ComplexElement< KSGenPositionFluxTube >( "position_flux_tube" );
}
