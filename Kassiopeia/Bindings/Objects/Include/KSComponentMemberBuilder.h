#ifndef Kassiopeia_KSComponentMemberBuilder_h_
#define Kassiopeia_KSComponentMemberBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentMember.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    class KSComponentMemberData
    {
        public:
            string fName;
            KSComponent* fParent;
            string fField;
    };

    typedef KComplexElement< KSComponentMemberData > KSComponentBuilder;

    template< >
    inline bool KSComponentBuilder::Begin()
    {
        fObject = new KSComponentMemberData;
        return true;
    }

    template< >
    inline bool KSComponentBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            string tName = aContainer->AsReference< string >();
            fObject->fName = tName;
            return true;
        }
        if( aContainer->GetName() == "parent" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            fObject->fParent = tComponent;
            return true;
        }
        if( aContainer->GetName() == "field" )
        {
            string tField = aContainer->AsReference< string >();
            fObject->fField = tField;
            return true;
        }
        return false;
    }

    template< >
    inline bool KSComponentBuilder::End()
    {
        KSComponent* tComponent = fObject->fParent->Component( fObject->fField );
        if( fObject->fName.size() != 0 )
        {
            tComponent->SetName( fObject->fName );
        }
        delete fObject;
        Set( tComponent );
        return true;
    }

}

#endif
