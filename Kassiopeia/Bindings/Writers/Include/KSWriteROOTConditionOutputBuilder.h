#ifndef Kassiopeia_KSWriteROOTConditionOutputBuilder_h_
#define Kassiopeia_KSWriteROOTConditionOutputBuilder_h_

#include "KComplexElement.hh"
#include "KSWriteROOTConditionOutput.h"
#include "KToolbox.h"
#include "KSComponentGroup.h"
#include <limits>

using namespace Kassiopeia;
namespace katrin
{

	class KSWriteROOTConditionOutputData
	{
		public:
			std::string fName;
			std::string fGroupName;
			std::string fComponentName;
			double fMinValue;
			double fMaxValue;
	};

    typedef KComplexElement< KSWriteROOTConditionOutputData > KSWriteROOTConditionOutputBuilder;

    template< >
    inline bool KSWriteROOTConditionOutputBuilder::Begin()
    {
        fObject = new KSWriteROOTConditionOutputData;
        fObject->fMinValue = -1.0*std::numeric_limits< double >::max();
        fObject->fMaxValue = std::numeric_limits< double >::max();
        return true;
    }

    template< >
    inline bool KSWriteROOTConditionOutputBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            std::string tName = aContainer->AsReference< std::string >();
            fObject->fName = tName;
            return true;
        }
        if( aContainer->GetName() == "min_value" )
        {
            double tValue = aContainer->AsReference< double >();
            fObject->fMinValue = tValue;
            return true;
        }
        if( aContainer->GetName() == "max_value" )
        {
            double tValue = aContainer->AsReference< double >();
            fObject->fMaxValue = tValue;
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
    inline bool KSWriteROOTConditionOutputBuilder::End()
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
                objctmsg( eError ) << "write ROOT condition output builder could not find component <" << fObject->fComponentName << "> in group <" << fObject->fGroupName << ">" << eom;
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
        	KSWriteROOTConditionOutput< unsigned short >* tWriteROOTConditionOutput = new KSWriteROOTConditionOutput< unsigned short >();
        	tWriteROOTConditionOutput->SetName( fObject->fName );
        	tWriteROOTConditionOutput->SetMinValue( fObject->fMinValue );
        	tWriteROOTConditionOutput->SetMaxValue( fObject->fMaxValue );
        	tWriteROOTConditionOutput->SetValue( tComponent->As< unsigned short >() );
        	tCondition = tWriteROOTConditionOutput;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< short >() == true )
        {
            KSWriteROOTConditionOutput< short >* tWriteROOTConditionOutput = new KSWriteROOTConditionOutput< short >();
            tWriteROOTConditionOutput->SetName( fObject->fName );
            tWriteROOTConditionOutput->SetMinValue( fObject->fMinValue );
            tWriteROOTConditionOutput->SetMaxValue( fObject->fMaxValue );
            tWriteROOTConditionOutput->SetValue( tComponent->As< short >() );
        	tCondition = tWriteROOTConditionOutput;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< unsigned int >() == true )
        {
            KSWriteROOTConditionOutput< unsigned int >* tWriteROOTConditionOutput = new KSWriteROOTConditionOutput< unsigned int >();
            tWriteROOTConditionOutput->SetName( fObject->fName );
            tWriteROOTConditionOutput->SetMinValue( fObject->fMinValue );
            tWriteROOTConditionOutput->SetMaxValue( fObject->fMaxValue );
            tWriteROOTConditionOutput->SetValue( tComponent->As< unsigned int >() );
        	tCondition = tWriteROOTConditionOutput;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< int >() == true )
        {
            KSWriteROOTConditionOutput< int >* tWriteROOTConditionOutput = new KSWriteROOTConditionOutput< int >();
            tWriteROOTConditionOutput->SetName( fObject->fName );
            tWriteROOTConditionOutput->SetMinValue( fObject->fMinValue );
            tWriteROOTConditionOutput->SetMaxValue( fObject->fMaxValue );
            tWriteROOTConditionOutput->SetValue( tComponent->As< int >() );
        	tCondition = tWriteROOTConditionOutput;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< unsigned long >() == true )
        {
            KSWriteROOTConditionOutput< unsigned long >* tWriteROOTConditionOutput = new KSWriteROOTConditionOutput< unsigned long >();
            tWriteROOTConditionOutput->SetName( fObject->fName );
            tWriteROOTConditionOutput->SetMinValue( fObject->fMinValue );
            tWriteROOTConditionOutput->SetMaxValue( fObject->fMaxValue );
            tWriteROOTConditionOutput->SetValue( tComponent->As< unsigned long >() );
        	tCondition = tWriteROOTConditionOutput;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< long >() == true )
        {
            KSWriteROOTConditionOutput< long >* tWriteROOTConditionOutput = new KSWriteROOTConditionOutput< long >();
            tWriteROOTConditionOutput->SetName( fObject->fName );
            tWriteROOTConditionOutput->SetMinValue( fObject->fMinValue );
            tWriteROOTConditionOutput->SetMaxValue( fObject->fMaxValue );
            tWriteROOTConditionOutput->SetValue( tComponent->As< long >() );
        	tCondition = tWriteROOTConditionOutput;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< float >() == true )
        {
            KSWriteROOTConditionOutput< float >* tWriteROOTConditionOutput = new KSWriteROOTConditionOutput< float >();
            tWriteROOTConditionOutput->SetName( fObject->fName );
            tWriteROOTConditionOutput->SetMinValue( fObject->fMinValue );
            tWriteROOTConditionOutput->SetMaxValue( fObject->fMaxValue );
            tWriteROOTConditionOutput->SetValue( tComponent->As< float >() );
        	tCondition = tWriteROOTConditionOutput;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< double >() == true )
        {
            KSWriteROOTConditionOutput< double >* tWriteROOTConditionOutput = new KSWriteROOTConditionOutput< double >();
            tWriteROOTConditionOutput->SetName( fObject->fName );
            tWriteROOTConditionOutput->SetMinValue( fObject->fMinValue );
            tWriteROOTConditionOutput->SetMaxValue( fObject->fMaxValue );
            tWriteROOTConditionOutput->SetValue( tComponent->As< double >() );
        	tCondition = tWriteROOTConditionOutput;
            delete fObject;
			Set( tCondition );
			return true;
        }

        objctmsg( eError ) << "component in write ROOT condition output builder is of non supported type " << eom;
        return false;
    }

}
#endif
