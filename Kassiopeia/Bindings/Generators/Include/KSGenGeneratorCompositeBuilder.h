#ifndef Kassiopeia_KSGenGeneratorCompositeBuilder_h_
#define Kassiopeia_KSGenGeneratorCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenGeneratorComposite.h"
#include "KSGeneratorsMessage.h"
#include "KSGenValueFix.h"
#include "KSGenStringValueFix.h"
#include "KToolbox.h"

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
        if( aContainer->GetName() == "energy" )
        {
            fObject->AddCreator( KToolbox::GetInstance().Get< KSGenCreator >( aContainer->AsReference< std::string >() ) );
            genmsg(eWarning) << "This option is deprecated and will be removed in the future. Use creator to add creators or nest the builders" << eom;
            return true;
        }
        if( aContainer->GetName() == "position" )
        {
            fObject->AddCreator( KToolbox::GetInstance().Get< KSGenCreator >( aContainer->AsReference< std::string >() ) );
            genmsg(eWarning) << "This option is deprecated and will be removed in the future. Use creator to add creators or nest the builders" << eom;
            return true;
        }
        if( aContainer->GetName() == "direction" )
        {
            fObject->AddCreator( KToolbox::GetInstance().Get< KSGenCreator >( aContainer->AsReference< std::string >() ) );
            genmsg(eWarning) << "This option is deprecated and will be removed in the future. Use creator to add creators or nest the builders" << eom;
            return true;
        }
        if( aContainer->GetName() == "time" )
        {
            fObject->AddCreator( KToolbox::GetInstance().Get< KSGenCreator >( aContainer->AsReference< std::string >() ) );
            genmsg(eWarning) << "This option is deprecated and will be removed in the future. Use creator to add creators or nest the builders" << eom;
            return true;
        }
        if( aContainer->GetName() == "creator" )
        {
            fObject->AddCreator( KToolbox::GetInstance().Get< KSGenCreator >( aContainer->AsReference< std::string >() ) );
            return true;
        }
        if( aContainer->GetName() == "special" )
        {
            fObject->AddSpecial( KToolbox::GetInstance().Get< KSGenSpecial >( aContainer->AsReference< std::string >() ) );
            return true;
        }
        if( aContainer->GetName() == "pid" )
        {
            KSGenValueFix* tPidValue = new KSGenValueFix();
            tPidValue->SetValue(aContainer->AsReference<double>());
            fObject->SetPid(tPidValue);
            return true;
        }
	if( aContainer->GetName() == "string_id" )
        {
	  KSGenStringValueFix* tStringIdValue = new KSGenStringValueFix();
	  tStringIdValue->SetValue(aContainer->AsReference<std::string>());
	  fObject->SetStringId(tStringIdValue);
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
        if(aContainer->Is<KSGenValue>() )
        {
            aContainer->ReleaseTo( fObject, &KSGenGeneratorComposite::SetPid );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSGenGeneratorCompositeBuilder::End()
    {
       if ((fObject->GetPid() != NULL) && (fObject->GetStringId() != NULL))
        {
            genmsg(eWarning) << "pid <" << fObject->GetPid() << "> overrides string_id <" << fObject->GetStringId() << ">. Only one should be used, to avoid confusion." << eom;
        }
       else if ((fObject->GetPid() == NULL) && (fObject->GetStringId() == NULL))
        {
            genmsg(eWarning) << "No particle id (pid) or string_id was set. Kassiopeia assumes that electrons (pid=11 or string_id=\"e-\") should be tracked" << eom;
            KSGenValueFix* tPidValue = new KSGenValueFix();
            tPidValue->SetValue(11);
            fObject->SetPid(tPidValue);
        }
       
       return true;
    }
    
}

#endif
