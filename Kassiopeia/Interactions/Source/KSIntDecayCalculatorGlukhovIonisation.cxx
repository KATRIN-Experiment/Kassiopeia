#include "KSIntDecayCalculatorGlukhovIonisation.h"
#include "KSInteractionsMessage.h"
#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSIntDecayCalculatorGlukhovIonisation::KSIntDecayCalculatorGlukhovIonisation() :
            fTargetPID( 0 ),
            fminPID( 0 ),
            fmaxPID( 0 ),
            fDecayProductGenerator( NULL )
    {
    }

    KSIntDecayCalculatorGlukhovIonisation::KSIntDecayCalculatorGlukhovIonisation( const KSIntDecayCalculatorGlukhovIonisation& aCopy ) :
            KSComponent(),
            fTargetPID( aCopy.fTargetPID ),
            fminPID( aCopy.fminPID ),
            fmaxPID( aCopy.fmaxPID ),
            fDecayProductGenerator( aCopy.fDecayProductGenerator )
    {
    }

    KSIntDecayCalculatorGlukhovIonisation* KSIntDecayCalculatorGlukhovIonisation::Clone() const
    {
        return new KSIntDecayCalculatorGlukhovIonisation( *this );
    }

    KSIntDecayCalculatorGlukhovIonisation::~KSIntDecayCalculatorGlukhovIonisation()
    {
    }

    const double KSIntDecayCalculatorGlukhovIonisation::low_n_rates[32] = { 57.32484,   146.49681,   286.6242 ,   452.2293 ,   630.57324,
                                                                        802.5478 ,   968.1529 ,  1101.9109 ,  1210.191  ,  1292.9937 ,
                                                                        1350.3185 ,  1394.9044 ,  1420.3822 ,  1433.121  ,  1433.121  ,
                                                                        1420.3822 ,  1407.6433 ,  1382.1656 ,  1356.6879 ,  1324.8408 ,
                                                                        1292.9937 ,  1267.5159 ,  1229.2993 ,  1191.0828 ,  1165.6051 ,
                                                                        1121.0192 ,  1089.172  ,  1050.9554 ,  1019.1083 ,   987.26117,
                                                                        961.78345,   923.5669 };

    const double KSIntDecayCalculatorGlukhovIonisation::p_coefficients[3][4] = { {5.8664e8,-0.3634 ,-16.704,51.07 },
                                                                       {5.4448e9,-0.03953,-1.5171,5.115 },
                                                                       {1.9153e9,-0.11334,-3.1140,12.913}
                                                                     };
    const double KSIntDecayCalculatorGlukhovIonisation::b_ion[3][3][3] = {{{13.9123, -21.8262, 10.2640},
                                                                           {-27.3489, 55.6957, -29.5954},       // s
                                                                           {12.9176, -26.3804, 13.6662}},

                                                                          {{1.4983,-2.3516,1.1062},
                                                                           {-3.1134,6.4271,-3.4492},            // p
                                                                           {1.4554,-3.0090,1.5756}},

                                                                          {{ 4.3096, -6.7807, 3.1954},
                                                                           { -8.4080, 17.2370, -9.2097},        // d
                                                                           { 3.8665, -7.9279, 4.1228}}};

    const double KSIntDecayCalculatorGlukhovIonisation::q_0 = 4./3.;
    const double KSIntDecayCalculatorGlukhovIonisation::A = 560./9.;
    const double KSIntDecayCalculatorGlukhovIonisation::B = 500./3.;


    void KSIntDecayCalculatorGlukhovIonisation::CalculateLifeTime( const KSParticle& aParticle,
                                                                      double& aLifeTime )
    {
        long long tPID = aParticle.GetPID();
        if ( (tPID == fTargetPID && fTargetPID != 0) || ( (tPID >= fminPID) && ( tPID <= fmaxPID)) )
        {
            int n = aParticle.GetMainQuantumNumber();
            // int l = aParticle.GetSecondQuantumNumber();

            if( n > 40) {
                aLifeTime = 1. / (CalculateSpontaneousDecayRate(n, 1)
                                  * CalculateRelativeIonisationRate(n, 1, fTemperature));
            }else {
                if(n >= 9) {
                    aLifeTime = 1. / low_n_rates[n - 9];
                } else
                {
                    intmsg( eError ) << "ionisation rate not implemented for n < 9" << eom;
                }

            }

        } else
        {
            aLifeTime = std::numeric_limits<double>::max();
        }
        return;
    }

    void KSIntDecayCalculatorGlukhovIonisation::ExecuteInteraction( const KSParticle& anInitialParticle,
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

    void KSIntDecayCalculatorGlukhovIonisation::SetDecayProductGenerator( KSGenerator* aGenerator )
    {
        if( fDecayProductGenerator == NULL )
        {
            fDecayProductGenerator = aGenerator;
            return;
        }
        intmsg( eError ) << "cannot set decay product generator <" << aGenerator->GetName() << "> to decay calculator <" << GetName() << ">" << eom;
        return;
    }

    KSGenerator* KSIntDecayCalculatorGlukhovIonisation::GetDecayProductGenerator() const
    {
        return fDecayProductGenerator;
    }

    double KSIntDecayCalculatorGlukhovIonisation::CalculateSpontaneousDecayRate( int n, int l)
    {
        if (l > 2 || l < 0 || n<0)
            intmsg(eError) << GetName() << ": Only l states 0,1,2 available for Rydberg decay" << eom;

        return p_coefficients[l][0]/(n*n*n)*(1.+p_coefficients[l][1]/n+p_coefficients[l][2]/(n*n)+p_coefficients[l][3]/(n*n*n));
    }

    double KSIntDecayCalculatorGlukhovIonisation::tau_tilde(double T)
    {
        return std::sqrt(100.0/T);
    }

    double KSIntDecayCalculatorGlukhovIonisation::a_ion(int l,int i,double T)
    {
        double tTauTilde = tau_tilde(T);
        return b_ion[l][i][0]+b_ion[l][i][1]*tTauTilde+b_ion[l][i][2]*tTauTilde*tTauTilde;
    }

    double KSIntDecayCalculatorGlukhovIonisation::q(double T)
    {
        return q_0-A/(T+B);
    }

    double KSIntDecayCalculatorGlukhovIonisation::y(int n, double T)
    {
        return 157890.0/(n*n*T);
    }

    double KSIntDecayCalculatorGlukhovIonisation::CalculateRelativeIonisationRate(int n, int l, double T)
    {
        if (l > 2 || l < 0 || n<0)
            intmsg(eError) << "KSIntDecayCalculatorGlukhovIonisation: Only l states 0,1,2 available for Rydberg decay" << eom;

        return (a_ion(l,0,T)+a_ion(l,1,T)*y(n,T)+a_ion(l,2,T)*y(n,T)*y(n,T))

                /(std::pow(n,q(T))*(std::exp(y(n,T))-1));

    }
}

