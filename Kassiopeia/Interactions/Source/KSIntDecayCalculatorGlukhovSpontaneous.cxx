#include "KSIntDecayCalculatorGlukhovSpontaneous.h"
#include "KSInteractionsMessage.h"
#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSIntDecayCalculatorGlukhovSpontaneous::KSIntDecayCalculatorGlukhovSpontaneous() :
            fTargetPID( 0 ),
            fminPID( 0 ),
            fmaxPID( 0 )
    {
    }

    KSIntDecayCalculatorGlukhovSpontaneous::KSIntDecayCalculatorGlukhovSpontaneous( const KSIntDecayCalculatorGlukhovSpontaneous& aCopy ) :
            KSComponent(),
            fTargetPID( aCopy.fTargetPID ),
            fminPID( aCopy.fminPID ),
            fmaxPID( aCopy.fmaxPID )
    {
    }

    KSIntDecayCalculatorGlukhovSpontaneous* KSIntDecayCalculatorGlukhovSpontaneous::Clone() const
    {
        return new KSIntDecayCalculatorGlukhovSpontaneous( *this );
    }

    KSIntDecayCalculatorGlukhovSpontaneous::~KSIntDecayCalculatorGlukhovSpontaneous()
    {
    }

    const double KSIntDecayCalculatorGlukhovSpontaneous::p_coefficients[3][4] = { {5.8664e8,-0.3634 ,-16.704,51.07 },
                                                                       {5.4448e9,-0.03953,-1.5171,5.115 },
                                                                       {1.9153e9,-0.11334,-3.1140,12.913}
                                                                     };

    void KSIntDecayCalculatorGlukhovSpontaneous::CalculateLifeTime( const KSParticle& aParticle,
                                                                      double& aLifeTime )
    {
        long long tPID = aParticle.GetPID();
        if ( (tPID == fTargetPID && fTargetPID != 0) || ( (tPID >= fminPID) && ( tPID <= fmaxPID)) )
        {
            int n = aParticle.GetMainQuantumNumber();
            int l = aParticle.GetSecondQuantumNumber();

            if( l == 0)
                aLifeTime = 1./CalculateSpontaneousDecayRate( n,l );
            else
                aLifeTime = 93.42e-12*n*n*n*l*(l+1);
        } else
        {
            aLifeTime = std::numeric_limits<double>::max();
        }
        return;
    }

    void KSIntDecayCalculatorGlukhovSpontaneous::ExecuteInteraction( const KSParticle& anInitialParticle,
                                                                 KSParticle& aFinalParticle,
                                                                 KSParticleQueue& /*aSecondaries*/)
    {
        aFinalParticle.SetTime( anInitialParticle.GetTime() );
        aFinalParticle.SetPosition( anInitialParticle.GetPosition() );
        aFinalParticle.SetMomentum( anInitialParticle.GetMomentum() );

        if ( (anInitialParticle.GetPID() == fTargetPID && fTargetPID != 0) || ( (anInitialParticle.GetPID() >= fminPID) && ( anInitialParticle.GetPID() <= fmaxPID)) )
        {                       
            aFinalParticle.SetLabel( GetName() );
            aFinalParticle.SetActive(false);

            fStepNDecays = 1;
            fStepEnergyLoss = 0.;

        }

        return;
    }

//    double KSIntDecayCalculatorGlukhovSpontaneous::CalculateSpontaneousDecayRate( const KSParticle& aParticle )
//    {
//        int n = aParticle.GetMainQuantumNumber();
//        int l = aParticle.GetSecondQuantumNumber();

//        if (l > 2 || l < 0)
//            intmsg(eError) << "KSIntDecayCalculatorGlukhovSpontaneous: Only l states 0,1,2 available for Rydberg decay" << eom;

//        return p_coefficients[l][0]/(n*n*n)*(1.+p_coefficients[l][1]/n+p_coefficients[l][2]/(n*n)+p_coefficients[l][3]/(n*n*n));
//    }

    double KSIntDecayCalculatorGlukhovSpontaneous::CalculateSpontaneousDecayRate( int n, int l)
    {
        if (l > 2 || l < 0 || n<0)
            intmsg(eError) << "KSIntDecayCalculatorGlukhovSpontaneous: Only l states 0,1,2 available for Rydberg decay" << eom;

        return p_coefficients[l][0]/(n*n*n*1.)*(1.+p_coefficients[l][1]/(n*1.)+p_coefficients[l][2]/(n*n*1.)+p_coefficients[l][3]/(n*n*n*1.));
    }

}
