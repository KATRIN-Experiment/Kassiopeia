#ifndef Kassiopeia_KSCommandMemberBuilder_h_
#define Kassiopeia_KSCommandMemberBuilder_h_

#include "KComplexElement.hh"
#include "KSCommandMember.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    class KSCommandMemberData
    {
        public:
            string fName;
            KSComponent* fParent;
            KSComponent* fChild;
            string fField;
    };

    typedef KComplexElement< KSCommandMemberData > KSCommandMemberBuilder;

    template< >
    inline bool KSCommandMemberBuilder::Begin()
    {
        fObject = new KSCommandMemberData;
        return true;
    }

    template< >
    inline bool KSCommandMemberBuilder::AddAttribute( KContainer* aContainer )
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
        if( aContainer->GetName() == "child" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            fObject->fChild = tComponent;
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
    inline bool KSCommandMemberBuilder::End()
    {
        KSCommand* tCommand = fObject->fParent->Command( fObject->fField, fObject->fChild );
        if( fObject->fName.length() != 0 )
        {
            tCommand->SetName( fObject->fName );
        }
        delete fObject;
        Set( tCommand );
        return true;
    }

}

#endif
