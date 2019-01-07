#ifndef Kassiopeia_KSWriteROOTConditionPeriodicBuilder_h_
#define Kassiopeia_KSWriteROOTConditionPeriodicBuilder_h_

#include "KComplexElement.hh"
#include "KSWriteROOTConditionPeriodic.h"
#include "KToolbox.h"
#include "KSComponentGroup.h"
#include <limits>

using namespace Kassiopeia;
namespace katrin
{

	class KSWriteROOTConditionPeriodicData
	{
		public:
			std::string fName;
			std::string fGroupName;
			std::string fComponentName;
			double fInitialMin;
			double fInitialMax;
			double fIncrement;
			double fResetMin;
			double fResetMax;
	};

    typedef KComplexElement< KSWriteROOTConditionPeriodicData > KSWriteROOTConditionPeriodicBuilder;

    template< >
    inline bool KSWriteROOTConditionPeriodicBuilder::Begin()
    {
        fObject = new KSWriteROOTConditionPeriodicData;
        fObject->fInitialMin = std::numeric_limits< double >::max(); //NOTE: defaults to always-off until reset
        fObject->fInitialMax = -1.0*std::numeric_limits< double >::max();
        return true;
    }

    template< >
    inline bool KSWriteROOTConditionPeriodicBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            std::string tName = aContainer->AsReference< std::string >();
            fObject->fName = tName;
            return true;
        }
        if( aContainer->GetName() == "initial_min" )
        {
            double tValue = aContainer->AsReference< double >();
            fObject->fInitialMin = tValue;
            return true;
        }
        if( aContainer->GetName() == "initial_max" )
        {
            double tValue = aContainer->AsReference< double >();
            fObject->fInitialMax = tValue;
            return true;
        }
		if( aContainer->GetName() == "increment" )
        {
            double tValue = aContainer->AsReference< double >();
            fObject->fIncrement = tValue;
            return true;
        }
		if( aContainer->GetName() == "reset_min" )
        {
            double tValue = aContainer->AsReference< double >();
            fObject->fResetMin = tValue;
            return true;
        }
        if( aContainer->GetName() == "reset_max" )
        {
            double tValue = aContainer->AsReference< double >();
            fObject->fResetMax = tValue;
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
    inline bool KSWriteROOTConditionPeriodicBuilder::End()
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
        	KSWriteROOTConditionPeriodic< unsigned short >* tWriteROOTConditionPeriodic = new KSWriteROOTConditionPeriodic< unsigned short >();
        	tWriteROOTConditionPeriodic->SetName( fObject->fName );
        	tWriteROOTConditionPeriodic->SetInitialMin( fObject->fInitialMin );
        	tWriteROOTConditionPeriodic->SetInitialMax( fObject->fInitialMax );
			tWriteROOTConditionPeriodic->SetIncrement( fObject->fIncrement );
			tWriteROOTConditionPeriodic->SetResetMin( fObject->fResetMin );
        	tWriteROOTConditionPeriodic->SetResetMax( fObject->fResetMax );
        	tWriteROOTConditionPeriodic->SetValue( tComponent->As< unsigned short >() );
        	tCondition = tWriteROOTConditionPeriodic;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< short >() == true )
        {
            KSWriteROOTConditionPeriodic< short >* tWriteROOTConditionPeriodic = new KSWriteROOTConditionPeriodic< short >();
            tWriteROOTConditionPeriodic->SetName( fObject->fName );
            tWriteROOTConditionPeriodic->SetInitialMin( fObject->fInitialMin );
            tWriteROOTConditionPeriodic->SetInitialMax( fObject->fInitialMax );
			tWriteROOTConditionPeriodic->SetIncrement( fObject->fIncrement );
			tWriteROOTConditionPeriodic->SetResetMin( fObject->fResetMin );
        	tWriteROOTConditionPeriodic->SetResetMax( fObject->fResetMax );
            tWriteROOTConditionPeriodic->SetValue( tComponent->As< short >() );
        	tCondition = tWriteROOTConditionPeriodic;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< unsigned int >() == true )
        {
            KSWriteROOTConditionPeriodic< unsigned int >* tWriteROOTConditionPeriodic = new KSWriteROOTConditionPeriodic< unsigned int >();
            tWriteROOTConditionPeriodic->SetName( fObject->fName );
            tWriteROOTConditionPeriodic->SetInitialMin( fObject->fInitialMin );
            tWriteROOTConditionPeriodic->SetInitialMax( fObject->fInitialMax );
			tWriteROOTConditionPeriodic->SetIncrement( fObject->fIncrement );
			tWriteROOTConditionPeriodic->SetResetMin( fObject->fResetMin );
        	tWriteROOTConditionPeriodic->SetResetMax( fObject->fResetMax );
            tWriteROOTConditionPeriodic->SetValue( tComponent->As< unsigned int >() );
        	tCondition = tWriteROOTConditionPeriodic;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< int >() == true )
        {
            KSWriteROOTConditionPeriodic< int >* tWriteROOTConditionPeriodic = new KSWriteROOTConditionPeriodic< int >();
            tWriteROOTConditionPeriodic->SetName( fObject->fName );
            tWriteROOTConditionPeriodic->SetInitialMin( fObject->fInitialMin );
            tWriteROOTConditionPeriodic->SetInitialMax( fObject->fInitialMax );
			tWriteROOTConditionPeriodic->SetIncrement( fObject->fIncrement );
			tWriteROOTConditionPeriodic->SetResetMin( fObject->fResetMin );
        	tWriteROOTConditionPeriodic->SetResetMax( fObject->fResetMax );
            tWriteROOTConditionPeriodic->SetValue( tComponent->As< int >() );
        	tCondition = tWriteROOTConditionPeriodic;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< unsigned long >() == true )
        {
            KSWriteROOTConditionPeriodic< unsigned long >* tWriteROOTConditionPeriodic = new KSWriteROOTConditionPeriodic< unsigned long >();
            tWriteROOTConditionPeriodic->SetName( fObject->fName );
            tWriteROOTConditionPeriodic->SetInitialMin( fObject->fInitialMin );
            tWriteROOTConditionPeriodic->SetInitialMax( fObject->fInitialMax );
			tWriteROOTConditionPeriodic->SetIncrement( fObject->fIncrement );
			tWriteROOTConditionPeriodic->SetResetMin( fObject->fResetMin );
        	tWriteROOTConditionPeriodic->SetResetMax( fObject->fResetMax );
            tWriteROOTConditionPeriodic->SetValue( tComponent->As< unsigned long >() );
        	tCondition = tWriteROOTConditionPeriodic;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< long >() == true )
        {
            KSWriteROOTConditionPeriodic< long >* tWriteROOTConditionPeriodic = new KSWriteROOTConditionPeriodic< long >();
            tWriteROOTConditionPeriodic->SetName( fObject->fName );
            tWriteROOTConditionPeriodic->SetInitialMin( fObject->fInitialMin );
            tWriteROOTConditionPeriodic->SetInitialMax( fObject->fInitialMax );
			tWriteROOTConditionPeriodic->SetIncrement( fObject->fIncrement );
			tWriteROOTConditionPeriodic->SetResetMin( fObject->fResetMin );
        	tWriteROOTConditionPeriodic->SetResetMax( fObject->fResetMax );
            tWriteROOTConditionPeriodic->SetValue( tComponent->As< long >() );
        	tCondition = tWriteROOTConditionPeriodic;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< float >() == true )
        {
            KSWriteROOTConditionPeriodic< float >* tWriteROOTConditionPeriodic = new KSWriteROOTConditionPeriodic< float >();
            tWriteROOTConditionPeriodic->SetName( fObject->fName );
            tWriteROOTConditionPeriodic->SetInitialMin( fObject->fInitialMin );
            tWriteROOTConditionPeriodic->SetInitialMax( fObject->fInitialMax );
			tWriteROOTConditionPeriodic->SetIncrement( fObject->fIncrement );
			tWriteROOTConditionPeriodic->SetResetMin( fObject->fResetMin );
        	tWriteROOTConditionPeriodic->SetResetMax( fObject->fResetMax );
            tWriteROOTConditionPeriodic->SetValue( tComponent->As< float >() );
        	tCondition = tWriteROOTConditionPeriodic;
            delete fObject;
			Set( tCondition );
			return true;
        }

        if( tComponent->Is< double >() == true )
        {
            KSWriteROOTConditionPeriodic< double >* tWriteROOTConditionPeriodic = new KSWriteROOTConditionPeriodic< double >();
            tWriteROOTConditionPeriodic->SetName( fObject->fName );
            tWriteROOTConditionPeriodic->SetInitialMin( fObject->fInitialMin );
            tWriteROOTConditionPeriodic->SetInitialMax( fObject->fInitialMax );
			tWriteROOTConditionPeriodic->SetIncrement( fObject->fIncrement );
			tWriteROOTConditionPeriodic->SetResetMin( fObject->fResetMin );
        	tWriteROOTConditionPeriodic->SetResetMax( fObject->fResetMax );
            tWriteROOTConditionPeriodic->SetValue( tComponent->As< double >() );
        	tCondition = tWriteROOTConditionPeriodic;
            delete fObject;
			Set( tCondition );
			return true;
        }

        objctmsg( eError ) << "component in write ROOT condition output builder is of non supported type " << eom;
        return false;
    }

}
#endif
