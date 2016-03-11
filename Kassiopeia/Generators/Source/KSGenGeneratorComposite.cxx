#include "KSGenGeneratorComposite.h"
#include "KSParticleFactory.h"
#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{

    KSGenGeneratorComposite::KSGenGeneratorComposite() :
            fPidValue( NULL ),
            fSpecials( 128 ),
            fCreators( 128 )
    {
    }
    KSGenGeneratorComposite::KSGenGeneratorComposite( const KSGenGeneratorComposite& aCopy ) :
            KSComponent(),
            fPidValue( aCopy.fPidValue ),
            fSpecials( aCopy.fSpecials ),
            fCreators( aCopy.fCreators )
    {
    }
    KSGenGeneratorComposite* KSGenGeneratorComposite::Clone() const
    {
        return new KSGenGeneratorComposite( *this );
    }
    KSGenGeneratorComposite::~KSGenGeneratorComposite()
    {
    }

    void KSGenGeneratorComposite::SetPid(KSGenValue *aPidValue )
    {
        fPidValue = aPidValue;
        return;
    }

    KSGenValue* KSGenGeneratorComposite::GetPid()
    {
        return fPidValue;
    }

    void KSGenGeneratorComposite::AddSpecial( KSGenSpecial* aSpecial )
    {
        if( fSpecials.AddElement( aSpecial ) != -1 )
        {
            return;
        }
        genmsg( eError ) << "cannot add special creator <" << aSpecial->GetName() << "> to composite generator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenGeneratorComposite::RemoveSpecial( KSGenSpecial* aSpecial )
    {
        if( fSpecials.RemoveElement( aSpecial ) != -1 )
        {
            return;
        }
        genmsg( eError ) << "cannot remove special creator <" << aSpecial->GetName() << "> from composite generator <" << this->GetName() << ">" << eom;
        return;
    }

    void KSGenGeneratorComposite::AddCreator( KSGenCreator* aCreator )
    {
        if( fCreators.AddElement( aCreator ) != -1 )
        {
            return;
        }
        genmsg( eError ) << "cannot add creator <" << aCreator->GetName() << "> to composite generator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenGeneratorComposite::RemoveCreator( KSGenCreator* aCreator )
    {
        if( fCreators.RemoveElement( aCreator ) != -1 )
        {
            return;
        }
        genmsg( eError ) << "cannot remove creator <" << aCreator->GetName() << "> from composite generator <" << this->GetName() << ">" << eom;
        return;
    }


    void KSGenGeneratorComposite::ExecuteGeneration( KSParticleQueue& aPrimaries )
    {
        vector< double > tPIDValues;
        vector< double >::iterator tPIDValueIt;
        fPidValue->DiceValue(tPIDValues);

        for( tPIDValueIt = tPIDValues.begin(); tPIDValueIt != tPIDValues.end(); tPIDValueIt++ )
        {
            KSParticle* tParticle = KSParticleFactory::GetInstance()->Create( std::floor(*tPIDValueIt) );
            tParticle->AddLabel( GetName() );
            aPrimaries.push_back( tParticle );
        }

        fCreators.ForEach( &KSGenCreator::Dice, &aPrimaries );
        fSpecials.ForEach( &KSGenSpecial::DiceSpecial, &aPrimaries );

        // check if particle state is valid
        KSParticleIt tParticleIt;
        for( tParticleIt = aPrimaries.begin(); tParticleIt != aPrimaries.end(); tParticleIt++ )
        {
            KSParticle* tParticle = *tParticleIt;
            if (! tParticle->IsValid())
            {
                tParticle->Print();
                genmsg( eError ) << "invalid particle state in generator <" << this->GetName() << ">" << eom;
            }
            tParticle->ResetFieldCaching();
        }

        return;
    }

    void KSGenGeneratorComposite::InitializeComponent()
    {
        if(fPidValue == NULL)
        {
            genmsg(eError) << "NO PID VALUE" << eom;
        }
        fPidValue->Initialize();
        fCreators.ForEach( &KSGenCreator::Initialize );
        fSpecials.ForEach( &KSGenSpecial::Initialize );
        return;
    }

    void KSGenGeneratorComposite::DeinitializeComponent()
    {
        fPidValue->Deinitialize();
        fCreators.ForEach( &KSGenCreator::Deinitialize );
        fSpecials.ForEach( &KSGenSpecial::Deinitialize );
        return;
    }

    STATICINT sKSGenGeneratorCompositeDict =
        KSDictionary< KSGenGeneratorComposite >::AddCommand( &KSGenGeneratorComposite::AddSpecial, &KSGenGeneratorComposite::RemoveSpecial, "add_special", "remove_special" )+
        KSDictionary< KSGenGeneratorComposite >::AddCommand( &KSGenGeneratorComposite::AddCreator, &KSGenGeneratorComposite::RemoveCreator, "add_creator", "remove_creator" );

}
