#include "KSIntCalculatorHydrogenBuilder.h"
#include "KSRootBuilder.h"
#include "KSIntCalculatorConstantBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    KSIntCalculatorHydrogenSet::KSIntCalculatorHydrogenSet() :
    		fName( "anonymous" ),
    		fElastic ( true ),
    		fExcitation( true ),
    		fIonisation( true ),
    		fMolecule( "hydrogen" )
    {
    }
    KSIntCalculatorHydrogenSet::~KSIntCalculatorHydrogenSet()
    {
        for( vector< KSIntCalculator* >::iterator tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++ )
        {
            delete (*tIt);
        }
    }


    void KSIntCalculatorHydrogenSet::AddCalculator( KSIntCalculator* aCalculator )
    {
    	KSToolbox::GetInstance()->AddObject( aCalculator );
        fCalculators.push_back( aCalculator );
        return;
    }

    void KSIntCalculatorHydrogenSet::ReleaseCalculators( KSIntScattering* aScattering )
    {
        for( vector< KSIntCalculator* >::iterator tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++ )
        {
            aScattering->AddCalculator( *tIt );
        }
        fCalculators.clear();
        return;
    }

    template< >
    KSIntCalculatorHydrogenSetBuilder::~KComplexElement()
    {
    }

    static int sKSIntCalculatorHydrogenStructure =
        KSIntCalculatorHydrogenSetBuilder::Attribute< string >( "name" ) +
        KSIntCalculatorHydrogenSetBuilder::Attribute< bool >( "elastic" ) +
        KSIntCalculatorHydrogenSetBuilder::Attribute< bool >( "excitation" ) +
        KSIntCalculatorHydrogenSetBuilder::Attribute< bool >( "ionisation" ) +
        KSIntCalculatorHydrogenSetBuilder::Attribute< string >( "molecule" );

}
