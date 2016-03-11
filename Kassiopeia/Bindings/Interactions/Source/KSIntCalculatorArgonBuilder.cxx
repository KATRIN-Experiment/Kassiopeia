#include "KSIntCalculatorArgonBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    KSIntCalculatorArgonSet::KSIntCalculatorArgonSet():
        fName( "anonymous" ),
        fSingleIonisation( true ),
        fDoubleIonisation( true ),
        fExcitation( true ),
        fElastic( true )
    {
    }

    KSIntCalculatorArgonSet::~KSIntCalculatorArgonSet()
    {
        for( vector< KSIntCalculator* >::iterator tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++ )
        {
            delete (*tIt);
        }
    }

    void KSIntCalculatorArgonSet::AddCalculator( KSIntCalculator* aCalculator )
    {
        KSToolbox::GetInstance()->AddObject( aCalculator );
        fCalculators.push_back( aCalculator );
        return;
    }

    void KSIntCalculatorArgonSet::ReleaseCalculators( KSIntScattering* aScattering )
    {
        for( vector< KSIntCalculator* >::iterator tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++ )
        {
            aScattering->AddCalculator( *tIt );
        }
        fCalculators.clear();
        return;
    }

    template< >
    KSIntCalculatorArgonSetBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntCalculatorArgonStructure =
        KSIntCalculatorArgonSetBuilder::Attribute< string >( "name" ) +
        KSIntCalculatorArgonSetBuilder::Attribute< bool >( "elastic" ) +
        KSIntCalculatorArgonSetBuilder::Attribute< bool >( "excitation" ) +
        KSIntCalculatorArgonSetBuilder::Attribute< bool >( "single_ionisation" ) +
        KSIntCalculatorArgonSetBuilder::Attribute< bool >( "double_ionisation" );
}
