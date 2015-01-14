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

        return tParticle;
    }
    int KSParticleFactory::Define( const long long& aPID, const double& aMass, const double& aCharge, const double& aMoment )
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
        tParticle->fMoment = aMoment;

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
    static int sGhostDefinition = KSParticleFactory::GetInstance()->Define( 0, 0., 0., 0. );

    //electron
    static int sElectronDefinition = KSParticleFactory::GetInstance()->Define( 11, KConst::M_el(), -1. * KConst::Q(), 0.5 );

    //positron
    static int sPositronDefinition = KSParticleFactory::GetInstance()->Define( -11, KConst::M_el(), KConst::Q(), 0.5 );

    //muon
    static int sMuMinusDefinition = KSParticleFactory::GetInstance()->Define( 12, KConst::M_mu(), -1 * KConst::Q(), 0.5 );

    //anti-muon
    static int sMuPlusDefinition = KSParticleFactory::GetInstance()->Define( -12, KConst::M_mu(), KConst::Q(), 0.5 );

    //proton
    static int sProtonDefinition = KSParticleFactory::GetInstance()->Define( 2212, KConst::M_prot(), KConst::Q(), 0.5 );

    //anti-proton
    static int sAntiProtonDefinition = KSParticleFactory::GetInstance()->Define( -2212, KConst::M_prot(), -1 * KConst::Q(), 0.5 );

    //neutron
    static int sNeutronDefinition = KSParticleFactory::GetInstance()->Define( 2112, KConst::M_neut(), 0., 0.5 );

}
