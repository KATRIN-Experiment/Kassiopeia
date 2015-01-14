#include "KSGenEnergyCompositeBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSRootBuilder.h"

#ifdef KASSIOPEIA_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#endif

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenEnergyCompositeBuilder::~KComplexElement()
    {
    }

    static int sKSGenEnergyCompositeStructure =
        KSGenEnergyCompositeBuilder::Attribute< string >( "name" ) +
        KSGenEnergyCompositeBuilder::Attribute< string >( "energy" ) +
        KSGenEnergyCompositeBuilder::ComplexElement< KSGenValueFix >( "energy_fix" ) +
        KSGenEnergyCompositeBuilder::ComplexElement< KSGenValueSet >( "energy_set" ) +
        KSGenEnergyCompositeBuilder::ComplexElement< KSGenValueList >( "energy_list" ) +
        KSGenEnergyCompositeBuilder::ComplexElement< KSGenValueUniform >( "energy_uniform" ) +
        KSGenEnergyCompositeBuilder::ComplexElement< KSGenValueGauss >( "energy_gauss" ) ;

    static int sKSGenEnergyComposite =
        KSRootBuilder::ComplexElement< KSGenEnergyComposite >( "ksgen_energy_composite" );

#ifdef KASSIOPEIA_USE_ROOT
    static int sKSGenEnergyCompositeStructureROOT =
            KSGenEnergyCompositeBuilder::ComplexElement< KSGenValueFormula >( "energy_formula" );
#endif

}
