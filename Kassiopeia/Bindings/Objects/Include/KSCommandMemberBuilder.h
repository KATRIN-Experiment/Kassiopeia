#ifndef Kassiopeia_KSCommandMemberBuilder_h_
#define Kassiopeia_KSCommandMemberBuilder_h_

#include "KComplexElement.hh"
#include "KSCommandMember.h"
#include "KSObjectsMessage.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    class KSCommandMemberData
    {
        public:
            std::string fName;
            std::string fFieldName;
            std::string fParentName;
            std::string fChildName;
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
            std::string tName = aContainer->AsReference< std::string >();
            fObject->fName = tName;
            return true;
        }
        if( aContainer->GetName() == "parent" )
        {
            std::string tParent = aContainer->AsReference< std::string >();
            fObject->fParentName = tParent;
            return true;
        }
        if( aContainer->GetName() == "child" )
        {
            std::string tChild = aContainer->AsReference< std::string >();
            fObject->fChildName = tChild;
            return true;
        }
        if( aContainer->GetName() == "field" )
        {
            std::string tField = aContainer->AsReference< std::string >();
            fObject->fFieldName = tField;
            return true;
        }
        return false;
    }

    template< >
    inline bool KSCommandMemberBuilder::End()
    {
        KSComponent* tParent = KToolbox::GetInstance().Get< KSComponent >( fObject->fParentName );
        if ( tParent == nullptr )
        {
            objctmsg( eError ) << "command member <" << fObject->fName << "> could not find parent <" << fObject->fParentName << ">" << eom;
        }
        KSComponent* tChild = KToolbox::GetInstance().Get< KSComponent >( fObject->fChildName );
        if ( tChild == nullptr )
        {
            objctmsg( eError ) << "command member <" << fObject->fName << "> could not find child <" << fObject->fChildName << ">" << eom;
        }
        KSCommand* tCommand = tParent->Command( fObject->fFieldName, tChild );
        if ( tCommand == nullptr )
        {
            objctmsg( eError ) << "command member <" << fObject->fName << "> could not find field <" << fObject->fFieldName << "> with parent <" << fObject->fParentName << ">" << eom;
        }
        if( ! fObject->fName.empty() )
        {
            tCommand->SetName( fObject->fName );
        }
        delete fObject;
        Set( tCommand );
        return true;
    }

}

#endif
