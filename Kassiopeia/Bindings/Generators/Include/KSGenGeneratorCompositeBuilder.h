#ifndef Kassiopeia_KSGenGeneratorCompositeBuilder_h_
#define Kassiopeia_KSGenGeneratorCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenGeneratorComposite.h"
#include "KSGeneratorsMessage.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenGeneratorComposite > KSGenGeneratorCompositeBuilder;

    template< >
    inline bool KSGenGeneratorCompositeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSGenGeneratorComposite::SetName );
            return true;
        }
        if( aContainer->GetName() == "pid" )
        {
            aContainer->CopyTo( fObject, &KSGenGeneratorComposite::SetPid );
            return true;
        }
        if( aContainer->GetName() == "energy" )
        {
            fObject->AddCreator( KSToolbox::GetInstance()->GetObjectAs< KSGenCreator >( aContainer->AsReference< string >() ) );
            genmsg(eWarning) << "This option is deprecated and will be removed in the future. Use creator to add creators or nest the builders" << eom;
            return true;
        }
        if( aContainer->GetName() == "position" )
        {
            fObject->AddCreator( KSToolbox::GetInstance()->GetObjectAs< KSGenCreator >( aContainer->AsReference< string >() ) );
            genmsg(eWarning) << "This option is deprecated and will be removed in the future. Use creator to add creators or nest the builders" << eom;
            return true;
        }
        if( aContainer->GetName() == "direction" )
        {
            fObject->AddCreator( KSToolbox::GetInstance()->GetObjectAs< KSGenCreator >( aContainer->AsReference< string >() ) );
            genmsg(eWarning) << "This option is deprecated and will be removed in the future. Use creator to add creators or nest the builders" << eom;
            return true;
        }
        if( aContainer->GetName() == "time" )
        {
            fObject->AddCreator( KSToolbox::GetInstance()->GetObjectAs< KSGenCreator >( aContainer->AsReference< string >() ) );
            genmsg(eWarning) << "This option is deprecated and will be removed in the future. Use creator to add creators or nest the builders" << eom;
            return true;
        }
        if( aContainer->GetName() == "creator" )
        {
            fObject->AddCreator( KSToolbox::GetInstance()->GetObjectAs< KSGenCreator >( aContainer->AsReference< string >() ) );
            return true;
        }
        if( aContainer->GetName() == "special" )
        {
            fObject->AddSpecial( KSToolbox::GetInstance()->GetObjectAs< KSGenSpecial >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSGenGeneratorCompositeBuilder::AddElement( KContainer* aContainer )
    {
        if(aContainer->Is<KSGenCreator>() )
        {
            aContainer->ReleaseTo( fObject, &KSGenGeneratorComposite::AddCreator );
            return true;
        }
        return false;
    }

}

#endif
