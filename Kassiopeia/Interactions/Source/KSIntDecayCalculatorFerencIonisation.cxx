//
// Created by trost on 03.06.15.
//

#include "KSIntDecayCalculatorFerencIonisation.h"


#include "KSIntDecayCalculatorGlukhovIonisation.h"
#include "KSInteractionsMessage.h"
#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSIntDecayCalculatorFerencIonisation::KSIntDecayCalculatorFerencIonisation() :
            fTargetPID( 0 ),
            fminPID( 0 ),
            fmaxPID( 0 ),
            fTemperature( 300. ),
            fDecayProductGenerator( NULL )
    {
        fCalculator = new RydbergCalculator();
    }

    KSIntDecayCalculatorFerencIonisation::KSIntDecayCalculatorFerencIonisation( const KSIntDecayCalculatorFerencIonisation& aCopy ) :
            KSComponent(),
            fTargetPID( aCopy.fTargetPID ),
            fminPID( aCopy.fminPID ),
            fmaxPID( aCopy.fmaxPID ),
            fTemperature( aCopy.fTemperature ),
            fDecayProductGenerator( aCopy.fDecayProductGenerator )
    {
        fCalculator = new RydbergCalculator();
        for(int n = 0; n <=150; n++)
        {
            for( int l = 0; l < n; l++)
            {
                low_n_lifetimes[n][l] = aCopy.low_n_lifetimes[n][l];
            }
        }
    }

    void KSIntDecayCalculatorFerencIonisation::InitializeComponent() {

        for(int n = 0; n <150; n++)
        {
            for( int l = 0; l < n+1; l++)
            {
                low_n_lifetimes[n][l] = 1./fCalculator->PBBRionization(fTemperature,n+1,l,1,1.e-6,16);
            }
        }

    }

    KSIntDecayCalculatorFerencIonisation* KSIntDecayCalculatorFerencIonisation::Clone() const
    {
        return new KSIntDecayCalculatorFerencIonisation( *this );
    }

    KSIntDecayCalculatorFerencIonisation::~KSIntDecayCalculatorFerencIonisation()
    {
        delete fCalculator;
    }


    void KSIntDecayCalculatorFerencIonisation::CalculateLifeTime( const KSParticle& aParticle,
                                                                        double& aLifeTime )
    {
        long long tPID = aParticle.GetPID();
        if ( (tPID == fTargetPID && fTargetPID != 0) || ( (tPID >= fminPID) && ( tPID <= fmaxPID)) )
        {
            int n = aParticle.GetMainQuantumNumber();
            int l = aParticle.GetSecondQuantumNumber();

            if( n > 150 )
            {
                aLifeTime = 1./fCalculator->PBBRionization(fTemperature,n,l,1.,1.e-6,16);
            }else {
                aLifeTime = low_n_lifetimes[n-1][l];
            }

        } else
        {
            aLifeTime = std::numeric_limits<double>::max();
        }
        return;
    }

    void KSIntDecayCalculatorFerencIonisation::ExecuteInteraction( const KSParticle& anInitialParticle,
                                                                         KSParticle& aFinalParticle,
                                                                         KSParticleQueue& aSecondaries)
    {
        aFinalParticle.SetTime( anInitialParticle.GetTime() );
        aFinalParticle.SetPosition( anInitialParticle.GetPosition() );
        aFinalParticle.SetMomentum( anInitialParticle.GetMomentum() );

        if ( (anInitialParticle.GetPID() == fTargetPID && fTargetPID != 0) || ( (anInitialParticle.GetPID() >= fminPID) && ( anInitialParticle.GetPID() <= fmaxPID)) )
        {
            double tTime = aFinalParticle.GetTime();
            KThreeVector tPosition = aFinalParticle.GetPosition();

            aFinalParticle.SetLabel( GetName() );
            aFinalParticle.SetActive(false);

            fStepNDecays = 1;
            fStepEnergyLoss = 0.;

            if(fDecayProductGenerator != NULL)
            {
                KSParticleQueue tDecayProducts;

                fDecayProductGenerator->ExecuteGeneration(tDecayProducts);

                KSParticleIt tParticleIt;
                for( tParticleIt = tDecayProducts.begin(); tParticleIt != tDecayProducts.end(); tParticleIt++ )
                {
                    (*tParticleIt)->SetTime((*tParticleIt)->GetTime()+tTime);
                    (*tParticleIt)->SetPosition((*tParticleIt)->GetPosition()+tPosition);
                }

                aSecondaries = tDecayProducts;
            }
        }

        return;
    }

    void KSIntDecayCalculatorFerencIonisation::SetDecayProductGenerator( KSGenerator* aGenerator )
    {
        if( fDecayProductGenerator == NULL )
        {
            fDecayProductGenerator = aGenerator;
            return;
        }
        intmsg( eError ) << "cannot set decay product generator <" << aGenerator->GetName() << "> to decay calculator <" << GetName() << ">" << eom;
        return;
    }

    KSGenerator* KSIntDecayCalculatorFerencIonisation::GetDecayProductGenerator() const
    {
        return fDecayProductGenerator;
    }

}

