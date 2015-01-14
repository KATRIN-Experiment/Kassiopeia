#include "KSGenDirectionSurfaceCompositeBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueAngleSphericalBuilder.h"
#include "KSRootBuilder.h"

#ifdef KASSIOPEIA_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#endif

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenDirectionSurfaceCompositeBuilder::~KComplexElement()
    {
    }

    static int sKSGenDirectionSurfaceCompositeStructure =
            KSGenDirectionSurfaceCompositeBuilder::Attribute< string >( "name" ) +
            KSGenDirectionSurfaceCompositeBuilder::Attribute< string >( "theta" ) +
            KSGenDirectionSurfaceCompositeBuilder::Attribute< string >( "phi" ) +
            KSGenDirectionSurfaceCompositeBuilder::Attribute< string >( "surfaces" ) +
            KSGenDirectionSurfaceCompositeBuilder::Attribute< bool >( "side" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueFix >( "theta_fix" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueSet >( "theta_set" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueList >( "theta_list" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueUniform >( "theta_uniform" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueGauss >( "theta_gauss" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueAngleSpherical >( "theta_spherical" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueFix >( "phi_fix" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueSet >( "phi_set" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueList >( "phi_list" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueUniform >( "phi_uniform" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueGauss >( "phi_gauss" );

    static int sKSGenDirectionSurfaceComposite =
            KSRootBuilder::ComplexElement< KSGenDirectionSurfaceComposite >( "ksgen_direction_surface_composite" );

#ifdef KASSIOPEIA_USE_ROOT
    static int sKSGenDirectionSurfaceCompositeStructureROOT =
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueFormula >( "theta_formula" ) +
            KSGenDirectionSurfaceCompositeBuilder::ComplexElement< KSGenValueFormula >( "phi_formula" );
#endif

}
