#include "KSIntScatteringBuilder.h"
#include "KSIntDensityConstantBuilder.h"
#include "KSIntCalculatorConstantBuilder.h"
#include "KSIntCalculatorHydrogenBuilder.h"
#include "KSIntCalculatorArgonBuilder.h"
#include "KESSElasticElsepa.h"
#include "KESSInelasticBetheFano.h"
#include "KESSInelasticPenn.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    KSIntCalculatorSet::KSIntCalculatorSet()
    {
    }
    KSIntCalculatorSet::~KSIntCalculatorSet()
    {
    }

    template< >
    KSIntScatteringBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntScattering =
        KSRootBuilder::ComplexElement< KSIntScattering >( "ksint_scattering" );

    STATICINT sKSIntScatteringStructure =
        KSIntScatteringBuilder::Attribute< string >( "name" ) +
        KSIntScatteringBuilder::Attribute< bool >( "split" ) +
        KSIntScatteringBuilder::Attribute< string >( "density" ) +
        KSIntScatteringBuilder::Attribute< string >( "calculator" ) +
        KSIntScatteringBuilder::Attribute< string >( "calculators" ) +
        KSIntScatteringBuilder::Attribute< double >( "enhancement" ) +
        KSIntScatteringBuilder::ComplexElement< KSIntDensityConstant >( "density_constant" ) +
        KSIntScatteringBuilder::ComplexElement< KSIntCalculatorConstant >( "calculator_constant" ) +
        KSIntScatteringBuilder::ComplexElement< KSIntCalculatorHydrogenSet >( "calculator_hydrogen" ) +
        KSIntScatteringBuilder::ComplexElement< KSIntCalculatorArgonSet >( "calculator_argon" ) +
        KSIntScatteringBuilder::ComplexElement< KESSElasticElsepa >( "elastic_elsepa" ) +
        KSIntScatteringBuilder::ComplexElement< KESSInelasticBetheFano >( "inelastic_bethefano" ) +
        KSIntScatteringBuilder::ComplexElement< KESSInelasticPenn >( "inelastic_penn" );

}
