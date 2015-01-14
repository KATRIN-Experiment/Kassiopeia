#include "KSGenGeneratorComposite.h"
#include "KSParticleFactory.h"
#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{

    KSGenGeneratorComposite::KSGenGeneratorComposite() :
            fPid( 11 ),
            fCreators( 128 ),
            fSpecials( 128 )
    {
    }
    KSGenGeneratorComposite::KSGenGeneratorComposite( const KSGenGeneratorComposite& aCopy ) :
            fPid( aCopy.fPid ),            
            fCreators( aCopy.fCreators ),
            fSpecials( aCopy.fSpecials )
    {
    }
    KSGenGeneratorComposite* KSGenGeneratorComposite::Clone() const
    {
        return new KSGenGeneratorComposite( *this );
    }
    KSGenGeneratorComposite::~KSGenGeneratorComposite()
    {
    }

    void KSGenGeneratorComposite::SetPid( const long long& aPid )
    {
        fPid = aPid;
        return;
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

    	KSParticle* tParticle = KSParticleFactory::GetInstance()->Create( fPid );
    	tParticle->AddLabel( GetName() );
        aPrimaries.push_back( tParticle );

        fCreators.ForEach( &KSGenCreator::Dice, &aPrimaries );
        fSpecials.ForEach( &KSGenSpecial::DiceSpecial, &aPrimaries );

        return;
    }

    void KSGenGeneratorComposite::InitializeComponent()
    {
        fCreators.ForEach( &KSGenCreator::Initialize );
        fSpecials.ForEach( &KSGenSpecial::Initialize );
        return;
    }

    void KSGenGeneratorComposite::DeinitializeComponent()
    {
        fCreators.ForEach( &KSGenCreator::Deinitialize );
        fSpecials.ForEach( &KSGenSpecial::Deinitialize );
        return;
    }

    static int sKSGenGeneratorCompositeDict =    
        KSDictionary< KSGenGeneratorComposite >::AddCommand( &KSGenGeneratorComposite::AddSpecial, &KSGenGeneratorComposite::RemoveSpecial, "add_special", "remove_special" )+
        KSDictionary< KSGenGeneratorComposite >::AddCommand( &KSGenGeneratorComposite::AddCreator, &KSGenGeneratorComposite::RemoveCreator, "add_creator", "remove_creator" );

}
