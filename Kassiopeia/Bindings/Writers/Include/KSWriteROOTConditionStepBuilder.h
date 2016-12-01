#ifndef Kassiopeia_KSWriteROOTConditionStepBuilder_h_
#define Kassiopeia_KSWriteROOTConditionStepBuilder_h_

#include "KComplexElement.hh"
#include "KSWriteROOTConditionStep.h"
#include "KToolbox.h"
#include "KSComponentGroup.h"
#include <limits>

using namespace Kassiopeia;
namespace katrin
{

	class KSWriteROOTConditionStepData
	{
		public:
			std::string fName;
			std::string fGroupName;
			std::string fComponentName;
			int fNthStepValue;
	};

    typedef KComplexElement< KSWriteROOTConditionStepData > KSWriteROOTConditionStepBuilder;

    template< >
    inline bool KSWriteROOTConditionStepBuilder::Begin()
    {
        fObject = new KSWriteROOTConditionStepData;
        fObject->fNthStepValue = 1;
        return true;
    }

    template< >
    inline bool KSWriteROOTConditionStepBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            std::string tName = aContainer->AsReference< std::string >();
            fObject->fName = tName;
            return true;
        }
        if( aContainer->GetName() == "nth_step" )
        {
            int tValue = aContainer->AsReference< int >();
            fObject->fNthStepValue = tValue;
            objctmsg( eDebug ) << "Initializing KSWriteROOTConditionStep, writing every "<< tValue <<"th step to ROOT file" << eom;
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
        return false;
    }

    template< >
    inline bool KSWriteROOTConditionStepBuilder::End()
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
                objctmsg( eError ) << "write ROOT condition step builder could not find component <" << fObject->fComponentName << "> in group <" << fObject->fGroupName << ">" << eom;
                return false;
            }
        }
        else
        {
            tComponent = KToolbox::GetInstance().Get< KSComponent >( fObject->fComponentName );
        }


        KSWriteROOTCondition* tCondition = NULL;

        if( tComponent->Is< unsigned short >() == true )
        {
        	KSWriteROOTConditionStep< unsigned short >* tWriteROOTConditionStep = new KSWriteROOTConditionStep< unsigned short >();
        	tWriteROOTConditionStep->SetName( fObject->fName );
        	tWriteROOTConditionStep->SetNthStepValue( fObject->fNthStepValue );
        	tWriteROOTConditionStep->SetValue( tComponent->As< unsigned short >() );
        	tCondition = tWriteROOTConditionStep;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< short >() == true )
        {
            KSWriteROOTConditionStep< short >* tWriteROOTConditionStep = new KSWriteROOTConditionStep< short >();
            tWriteROOTConditionStep->SetName( fObject->fName );
            tWriteROOTConditionStep->SetNthStepValue( fObject->fNthStepValue );
            tWriteROOTConditionStep->SetValue( tComponent->As< short >() );
        	tCondition = tWriteROOTConditionStep;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< unsigned int >() == true )
        {
            KSWriteROOTConditionStep< unsigned int >* tWriteROOTConditionStep = new KSWriteROOTConditionStep< unsigned int >();
            tWriteROOTConditionStep->SetName( fObject->fName );
            tWriteROOTConditionStep->SetNthStepValue( fObject->fNthStepValue );
            tWriteROOTConditionStep->SetValue( tComponent->As< unsigned int >() );
        	tCondition = tWriteROOTConditionStep;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< int >() == true )
        {
            KSWriteROOTConditionStep< int >* tWriteROOTConditionStep = new KSWriteROOTConditionStep< int >();
            tWriteROOTConditionStep->SetName( fObject->fName );
            tWriteROOTConditionStep->SetNthStepValue( fObject->fNthStepValue );
            tWriteROOTConditionStep->SetValue( tComponent->As< int >() );
        	tCondition = tWriteROOTConditionStep;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< unsigned long >() == true )
        {
            KSWriteROOTConditionStep< unsigned long >* tWriteROOTConditionStep = new KSWriteROOTConditionStep< unsigned long >();
            tWriteROOTConditionStep->SetName( fObject->fName );
            tWriteROOTConditionStep->SetNthStepValue( fObject->fNthStepValue );
            tWriteROOTConditionStep->SetValue( tComponent->As< unsigned long >() );
        	tCondition = tWriteROOTConditionStep;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< long >() == true )
        {
            KSWriteROOTConditionStep< long >* tWriteROOTConditionStep = new KSWriteROOTConditionStep< long >();
            tWriteROOTConditionStep->SetName( fObject->fName );
            tWriteROOTConditionStep->SetNthStepValue( fObject->fNthStepValue );
            tWriteROOTConditionStep->SetValue( tComponent->As< long >() );
        	tCondition = tWriteROOTConditionStep;
            delete fObject;
			Set( tCondition );
			return true;
        }

        objctmsg( eError ) << "component in write ROOT condition step builder is of non supported type " << eom;
        return false;
    }

}
#endif
