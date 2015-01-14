#include "KSGenDirectionSphericalCompositeBuilder.h"
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
    KSGenDirectionSphericalCompositeBuilder::~KComplexElement()
    {
    }

    static int sKSGenDirectionSphericalCompositeStructure =
        KSGenDirectionSphericalCompositeBuilder::Attribute< string >( "name" ) +
        KSGenDirectionSphericalCompositeBuilder::Attribute< string >( "theta" ) +
        KSGenDirectionSphericalCompositeBuilder::Attribute< string >( "phi" ) +
        KSGenDirectionSphericalCompositeBuilder::Attribute< string >( "surface" ) +
        KSGenDirectionSphericalCompositeBuilder::Attribute< string >( "space" ) +
        KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueFix >( "theta_fix" ) +
        KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueSet >( "theta_set" ) +
        KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueList >( "theta_list" ) +
        KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueUniform >( "theta_uniform" ) +
        KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueGauss >( "theta_gauss" ) +
        KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueAngleSpherical >( "theta_spherical" ) +
        KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueFix >( "phi_fix" ) +
        KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueSet >( "phi_set" ) +
        KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueList >( "phi_list" ) +
        KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueUniform >( "phi_uniform" ) +
        KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueGauss >( "phi_gauss" );

    static int sKSGenDirectionSphericalComposite =
        KSRootBuilder::ComplexElement< KSGenDirectionSphericalComposite >( "ksgen_direction_spherical_composite" );

#ifdef KASSIOPEIA_USE_ROOT
    static int sKSGenDirectionSphericalCompositeStructureROOT =
            KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueFormula >( "theta_formula" ) +
            KSGenDirectionSphericalCompositeBuilder::ComplexElement< KSGenValueFormula >( "phi_formula" );
#endif

}
