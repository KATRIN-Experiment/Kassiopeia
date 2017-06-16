#ifndef Kassiopeia_KSComponentMinimumBuilder_h_
#define Kassiopeia_KSComponentMinimumBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentMinimum.h"
#include "KToolbox.h"
#include "KSObjectsMessage.h"
#include "KSComponentGroup.h"

#include "KTwoVector.hh"
using KGeoBag::KTwoVector;

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KTwoMatrix.hh"
using KGeoBag::KTwoMatrix;

#include "KThreeMatrix.hh"
using KGeoBag::KThreeMatrix;

using namespace Kassiopeia;
namespace katrin
{

    class KSComponentMinimumData
    {
        public:
            std::string fName;
            std::string fGroupName;
            std::string fParentName;
    };

    KSComponent* BuildOutputMinimum( KSComponent* aComponent )
    {
        if( aComponent->Is< bool >() == true )
        {
            return new KSComponentMinimum< bool >( aComponent, aComponent->As< bool >() );
        }

        if( aComponent->Is< unsigned char >() == true )
        {
            return new KSComponentMinimum< unsigned char >( aComponent, aComponent->As< unsigned char >() );
        }

        if( aComponent->Is< char >() == true )
        {
            return new KSComponentMinimum< char >( aComponent, aComponent->As< char >() );
        }

        if( aComponent->Is< unsigned short >() == true )
        {
            return new KSComponentMinimum< unsigned short >( aComponent, aComponent->As< unsigned short >() );
        }

        if( aComponent->Is< short >() == true )
        {
            return new KSComponentMinimum< short >( aComponent, aComponent->As< short >() );
        }

        if( aComponent->Is< unsigned int >() == true )
        {
            return new KSComponentMinimum< unsigned int >( aComponent, aComponent->As< unsigned int >() );
        }

        if( aComponent->Is< int >() == true )
        {
            return new KSComponentMinimum< int >( aComponent, aComponent->As< int >() );
        }

        if( aComponent->Is< unsigned long >() == true )
        {
            return new KSComponentMinimum< unsigned long >( aComponent, aComponent->As< unsigned long >() );
        }

        if( aComponent->Is< long >() == true )
        {
            return new KSComponentMinimum< long >( aComponent, aComponent->As< long >() );
        }

        if( aComponent->Is< float >() == true )
        {
            return new KSComponentMinimum< float >( aComponent, aComponent->As< float >() );
        }

        if( aComponent->Is< double >() == true )
        {
            return new KSComponentMinimum< double >( aComponent, aComponent->As< double >() );
        }

        if( aComponent->Is< KTwoVector >() == true )
        {
            return new KSComponentMinimum< KTwoVector >( aComponent, aComponent->As< KTwoVector >() );
        }

        if( aComponent->Is< KThreeVector >() == true )
        {
            return new KSComponentMinimum< KThreeVector >( aComponent, aComponent->As< KThreeVector >() );
        }

        if( aComponent->Is< KTwoMatrix >() == true )
        {
            return new KSComponentMinimum< KTwoMatrix >( aComponent, aComponent->As< KTwoMatrix >() );
        }

        if( aComponent->Is< KThreeMatrix >() == true )
        {
            return new KSComponentMinimum< KThreeMatrix >( aComponent, aComponent->As< KThreeMatrix >() );
        }

        return NULL;
    }

    typedef KComplexElement< KSComponentMinimumData > KSComponentMinimumBuilder;

    template< >
    inline bool KSComponentMinimumBuilder::Begin()
    {
        fObject = new KSComponentMinimumData;
        return true;
    }

    template< >
    inline bool KSComponentMinimumBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            std::string tName = aContainer->AsReference< std::string >();
            fObject->fName = tName;
            return true;
        }
        if( aContainer->GetName() == "group" )
        {
            std::string tGroupName = aContainer->AsReference< std::string >();
            fObject->fGroupName = tGroupName;
            return true;
        }
        if( aContainer->GetName() == "component" )
        {
            objctmsg( eWarning ) <<"deprecated warning in KSComponentMinimumBuilder: Please use the attribute <parent> instead <component>"<<eom;
            std::string tParentName = aContainer->AsReference< std::string >();
            fObject->fParentName = tParentName;
            return true;
        }
        if(  aContainer->GetName() == "parent" )
        {
            std::string tParentName = aContainer->AsReference< std::string >();
            fObject->fParentName = tParentName;
            return true;
        }
        return false;
    }

    template< >
    inline bool KSComponentMinimumBuilder::End()
    {
        KSComponent* tParentComponent = NULL;
        if( fObject->fGroupName.empty() == false )
        {
            KSComponentGroup* tComponentGroup = KToolbox::GetInstance().Get< KSComponentGroup >( fObject->fGroupName );
            for( unsigned int tIndex = 0; tIndex < tComponentGroup->ComponentCount(); tIndex++ )
            {
                KSComponent* tGroupComponent = tComponentGroup->ComponentAt( tIndex );
                if( tGroupComponent->GetName() == fObject->fParentName )
                {
                    tParentComponent = tGroupComponent;
                    break;
                }
            }
            if( tParentComponent == NULL )
            {
                objctmsg( eError ) << "component minimum builder could not find component <" << fObject->fParentName << "> in group <" << fObject->fGroupName << ">" << eom;
                return false;
            }
        }
        else
        {
            tParentComponent = KToolbox::GetInstance().Get< KSComponent >( fObject->fParentName );
        }
        KSComponent* tComponent = BuildOutputMinimum( tParentComponent );
        tComponent->SetName( fObject->fName );
        delete fObject;
        Set( tComponent );
        return true;
    }

}

#endif
