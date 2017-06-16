//
// Created by trost on 07.03.16.
//

#ifndef KASPER_KSWRITEROOTCONDITIONTERMINATORBUILDER_H
#define KASPER_KSWRITEROOTCONDITIONTERMINATORBUILDER_H
#include "KComplexElement.hh"
#include "KSWriteROOTConditionTerminator.h"
#include "KToolbox.h"
#include "KSComponentGroup.h"


using namespace Kassiopeia;
namespace katrin
{

class KSWriteROOTConditionTerminatorData
{
public:
    std::string fName;
    std::string fGroupName;
    std::string fComponentName;
    std::string fMatchTerminator;
};

typedef KComplexElement< KSWriteROOTConditionTerminatorData > KSWriteROOTConditionTerminatorBuilder;

template< >
inline bool KSWriteROOTConditionTerminatorBuilder::Begin()
{
    fObject = new KSWriteROOTConditionTerminatorData;
    return true;
}

template< >
inline bool KSWriteROOTConditionTerminatorBuilder::AddAttribute( KContainer* aContainer )
{
    if( aContainer->GetName() == "name" )
    {
        std::string tName = aContainer->AsReference< std::string >();
        fObject->fName = tName;
        return true;
    }
    if( aContainer->GetName() == "group" )
    {
        std::string tName = aContainer->AsReference< std::string >();
        fObject->fGroupName = tName;
        return true;
    }
    if( aContainer->GetName() == "parent" )
    {
        std::string tName = aContainer->AsReference< std::string >();
        fObject->fComponentName = tName;
        return true;
    }
    if( aContainer->GetName() == "match_terminator" )
    {
        std::string tMatchTerm = aContainer->AsReference< std::string >();
        fObject->fMatchTerminator = tMatchTerm;
        return true;
    }
    return false;
}

template< >
inline bool KSWriteROOTConditionTerminatorBuilder::End()
{
    KSComponent* tComponent = NULL;
    if( fObject->fGroupName.empty() == false )
    {
        KSComponentGroup* tComponentGroup = KToolbox::GetInstance().Get< KSComponentGroup >( fObject->fGroupName );
        for( unsigned int tIndex = 0; tIndex < tComponentGroup->ComponentCount(); tIndex++ )
        {
            KSComponent* tGroupComponent = tComponentGroup->ComponentAt( tIndex );
            if( tGroupComponent->GetName() == fObject->fComponentName )
            {
                tComponent = tGroupComponent;
                break;
            }
        }
        if( tComponent == NULL )
        {
            objctmsg( eError ) << "write ROOT condition terminator builder could not find component <" << fObject->fComponentName << "> in group <" << fObject->fGroupName << ">" << eom;
            return false;
        }
    }
    else
    {
        tComponent = KToolbox::GetInstance().Get< KSComponent >( fObject->fComponentName );
    }

    KSWriteROOTCondition* tCondition = NULL;

    KSWriteROOTConditionTerminator* tWriteROOTConditionTerminator = new KSWriteROOTConditionTerminator();
    tWriteROOTConditionTerminator->SetName( fObject->fName );
    tWriteROOTConditionTerminator->SetValue( tComponent->As< std::string >() );
    tWriteROOTConditionTerminator->SetMatchTerminator( fObject->fMatchTerminator );
    tCondition = tWriteROOTConditionTerminator;
    delete fObject;
    Set( tCondition );
    return true;
}

}

#endif //KASPER_KSWRITEROOTCONDITIONTERMINATORBUILDER_H
