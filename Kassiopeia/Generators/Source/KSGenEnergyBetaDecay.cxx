//
// Created by trost on 29.05.15.
//

#include "KSGenEnergyBetaDecay.h"
#include "KSGeneratorsMessage.h"

//#include "KSParticleFactory.h"
#include "KRandom.h"

using katrin::KRandom;

namespace Kassiopeia
{

    KSGenEnergyBetaDecay::KSGenEnergyBetaDecay() :
            fZDaughter( 2 ),
            fnmax( 1000 ),
            fFermiMax( 0. ),
            fEndpoint( 18600.),
            fmnu(0.),
            fMinEnergy( 0. ),
            fMaxEnergy( -1. )
    {
    }
    KSGenEnergyBetaDecay::KSGenEnergyBetaDecay( const KSGenEnergyBetaDecay& aCopy ) :
            KSComponent(),
            fZDaughter( aCopy.fZDaughter ),
            fnmax( aCopy.fnmax ),
            fFermiMax( aCopy.fFermiMax ),
            fEndpoint( aCopy.fEndpoint ),
            fmnu( aCopy.fmnu ),
            fMinEnergy( aCopy.fMinEnergy ),
            fMaxEnergy( aCopy.fMaxEnergy )
    {
    }
    KSGenEnergyBetaDecay* KSGenEnergyBetaDecay::Clone() const
    {
        return new KSGenEnergyBetaDecay( *this );
    }
    KSGenEnergyBetaDecay::~KSGenEnergyBetaDecay()
    {
    }

    void KSGenEnergyBetaDecay::Dice( KSParticleQueue* aPrimaries )
    {
        KSParticleIt tParticleIt;

        for( tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++ )
        {
            double tEnergy;
            do{
                tEnergy = GenBetaEnergy(fEndpoint,fmnu,fFermiMax,fZDaughter);
            }while(( tEnergy < fMinEnergy) || (tEnergy > fMaxEnergy));

            (*tParticleIt)->SetKineticEnergy_eV( tEnergy );
            (*tParticleIt)->SetLabel( GetName() );
        }


        return;
    }

    double KSGenEnergyBetaDecay::Fermi(double E,double mnu, double E0, double Z)
    {
        // This subr. computes the Fermi beta energy spectrum shape,
        // with neutrino mass
        //   E: electron kinetic energy in eV
        //  mnu: neutrino mass
        double beta,E1,p1,p2,E2,FC,x,Fermiret;
        E2=KConst::M_el_eV()+E;  // electron total energy
        p2=sqrt(E2*E2-KConst::M_el_eV()*KConst::M_el_eV());  // electron momentum
        beta=p2/E2;
        E1=E0-E;  // neutrino total energy
        if(E1>=mnu)
            p1=sqrt(fabs(E1*E1-mnu*mnu));  // neutrino momentum
        else
            p1=0.;
        x=2.*KConst::Pi()*Z*KConst::Alpha()/beta;
        FC=x/(1.-exp(-x));  // Coulomb correction factor
        Fermiret=p2*E2*p1*E1*FC;
        return Fermiret;
    }

    double KSGenEnergyBetaDecay::GetFermiMax(double E0, double mnu, double Z)
    {
        double Fermimax=0;

        for(int i=0;i<fnmax;i++){
            double E = (E0 / double (fnmax) )*i;
            double F = Fermi(E,mnu,E0,Z) ;
            if( F > Fermimax ){
                Fermimax = F;
            }
        }
        //Fermimax*=1.05;
        return Fermimax;
    }

    double KSGenEnergyBetaDecay::GenBetaEnergy(double E0, double mnu, double Fermimax, double Z)
    {
        // Generation of E:
        double E = 0;
        double F = 0;

        do{

            E = E0 * KRandom::GetInstance().Uniform(fMinEnergy/E0, fMaxEnergy/E0);
            F = Fermi(E,mnu,E0,Z);

        }while((Fermimax * KRandom::GetInstance().Uniform() ) > F);

        return E;
    }


    void KSGenEnergyBetaDecay::InitializeComponent()
    {
        fFermiMax = GetFermiMax(fEndpoint,fmnu,fZDaughter);
        if ( fMaxEnergy == -1. )
            fMaxEnergy = fEndpoint;


        return;
    }
    void KSGenEnergyBetaDecay::DeinitializeComponent()
    {
        return;
    }

}
