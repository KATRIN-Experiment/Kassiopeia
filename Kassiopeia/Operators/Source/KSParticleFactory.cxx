#include "KSParticleFactory.h"
#include "KSOperatorsMessage.h"

namespace Kassiopeia
{

    KSParticleFactory::KSParticleFactory() :
        fParticles(),
        fSpace( NULL ),
        fMagneticField( NULL ),
        fElectricField( NULL )
    {
    }
    KSParticleFactory::~KSParticleFactory()
    {
    }

    KSParticle* KSParticleFactory::Create( const long long& aPID )
    {
        ParticleCIt tIter = fParticles.find( aPID );
        if( tIter == fParticles.end() )
        {
            tIter = fParticles.find( 0 );
            oprmsg( eError ) << "could not find particle for pid <" << aPID << ">" << eom;
        }

        KSParticle* tParticle = new KSParticle( *(tIter->second) );
        tParticle->SetCurrentSpace( fSpace );
        tParticle->SetMagneticFieldCalculator( fMagneticField );
        tParticle->SetElectricFieldCalculator( fElectricField );
        tParticle->RecalculateSpinBody();

        return tParticle;
    }
    int KSParticleFactory::Define( const long long& aPID, const double& aMass, const double& aCharge, const double& aSpinMagnitude, const double& aGyromagneticRatio  )
    {
        ParticleIt tIter = fParticles.find( aPID );
        if( tIter != fParticles.end() )
        {
            oprmsg( eError ) << "asked to add definition for pid <" << aPID << "> with one already defined" << eom;
            return -1;
        }

        KSParticle* tParticle = new KSParticle();
        tParticle->fPID = aPID;
        tParticle->fMass = aMass;
        tParticle->fCharge = aCharge;
        tParticle->fSpinMagnitude = aSpinMagnitude;
        tParticle->fGyromagneticRatio = aGyromagneticRatio;

        fParticles.insert( ParticleEntry( aPID, tParticle ) );

        return 0;
    }

    void KSParticleFactory::SetSpace( KSSpace* aSpace )
    {
        fSpace = aSpace;
        return;
    }
    KSSpace* KSParticleFactory::GetSpace()
    {
        return fSpace;
    }


    void KSParticleFactory::SetMagneticField( KSMagneticField* aMagneticField )
    {
        fMagneticField = aMagneticField;
        return;
    }
    KSMagneticField* KSParticleFactory::GetMagneticField()
    {
        return fMagneticField;
    }

    void KSParticleFactory::SetElectricField( KSElectricField* anElectricField )
    {
        fElectricField = anElectricField;
        return;
    }
    KSElectricField* KSParticleFactory::GetElectricField()
    {
        return fElectricField;
    }


    // A "ghost" particle
    STATICINT sGhostDefinition = KSParticleFactory::GetInstance().Define( 0, 0., 0., 0., 0. );

    //electron
    STATICINT sElectronDefinition = KSParticleFactory::GetInstance().Define( 11, KConst::M_el_kg(), -1. * KConst::Q(), 0.5, -1.760859644e+11 );

    //positron
    STATICINT sPositronDefinition = KSParticleFactory::GetInstance().Define( -11, KConst::M_el_kg(), KConst::Q(), 0.5, -1.760859644e+11 );

    //muon
    STATICINT sMuMinusDefinition = KSParticleFactory::GetInstance().Define( 12, KConst::M_mu_kg(), -1 * KConst::Q(), 0.5, -2.43318710e+7 );

    //anti-muon
    STATICINT sMuPlusDefinition = KSParticleFactory::GetInstance().Define( -12, KConst::M_mu_kg(), KConst::Q(), 0.5, -2.43318710e+7 );

    //proton
    STATICINT sProtonDefinition = KSParticleFactory::GetInstance().Define( 2212, KConst::M_prot_kg(), KConst::Q(), 0.5, 2.675221900e+8 );

    //anti-proton
    STATICINT sAntiProtonDefinition = KSParticleFactory::GetInstance().Define( -2212, KConst::M_prot_kg(), -1 * KConst::Q(), 0.5, 2.675221900e+8 );

    //neutron
    STATICINT sNeutronDefinition = KSParticleFactory::GetInstance().Define( 2112, KConst::M_neut_kg(), 0., 0.5, -1.83247172e+8 );

//NOTE: Still need values for tritium

    //Tritium triplet state
    STATICINT sTTripletDefinition = KSParticleFactory::GetInstance().Define( 10002, KConst::M_tPlus_kg() + KConst::M_el_kg(), 0, 0.5, -1.76e+11 );

    //T+
    STATICINT sTPlusDefinition = KSParticleFactory::GetInstance().Define( 31, KConst::M_tPlus_kg(), KConst::Q(), 0.5, 2.853493e+8 );

    //T3+
    STATICINT sT3PlusDefinition = KSParticleFactory::GetInstance().Define( 33, KConst::M_T2_kg()+KConst::M_tPlus_kg(), 1*KConst::Q(), 0.5, 0 );

     //T5+
    STATICINT sT5PlusDefinition = KSParticleFactory::GetInstance().Define( 35, 2*KConst::M_T2_kg()+KConst::M_tPlus_kg(), 1*KConst::Q(), 0.5, 0 );

    //T-
    STATICINT sTMinusDefinition = KSParticleFactory::GetInstance().Define( -31, KConst::M_tPlus_kg(), -1*KConst::Q(), 0.5, 0 );

    //rydberg states
    STATICINT sRydbergDefinition_0 = KSParticleFactory::GetInstance().Define( 10000, 1.008*KConst::AtomicMassUnit_kg(), 0., 0.5, 2.67513e+6 );

}
