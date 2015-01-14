#ifndef Kassiopeia_KSComponentIntegralBuilder_h_
#define Kassiopeia_KSComponentIntegralBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentIntegral.h"
#include "KSToolbox.h"
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

    class KSComponentIntegralData
    {
        public:
            string fName;
            string fGroupName;
            string fParentName;
    };

    KSComponent* BuildOutputIntegral( KSComponent* aComponent )
    {
        if( aComponent->Is< bool >() == true )
        {
            return new KSComponentIntegral< bool >( aComponent, aComponent->As< bool >() );
        }

        if( aComponent->Is< unsigned char >() == true )
        {
            return new KSComponentIntegral< unsigned char >( aComponent, aComponent->As< unsigned char >() );
        }

        if( aComponent->Is< char >() == true )
        {
            return new KSComponentIntegral< char >( aComponent, aComponent->As< char >() );
        }

        if( aComponent->Is< unsigned short >() == true )
        {
            return new KSComponentIntegral< unsigned short >( aComponent, aComponent->As< unsigned short >() );
        }

        if( aComponent->Is< short >() == true )
        {
            return new KSComponentIntegral< short >( aComponent, aComponent->As< short >() );
        }

        if( aComponent->Is< unsigned int >() == true )
        {
            return new KSComponentIntegral< unsigned int >( aComponent, aComponent->As< unsigned int >() );
        }

        if( aComponent->Is< int >() == true )
        {
            return new KSComponentIntegral< int >( aComponent, aComponent->As< int >() );
        }

        if( aComponent->Is< unsigned long >() == true )
        {
            return new KSComponentIntegral< unsigned long >( aComponent, aComponent->As< unsigned long >() );
        }

        if( aComponent->Is< long >() == true )
        {
            return new KSComponentIntegral< long >( aComponent, aComponent->As< long >() );
        }

        if( aComponent->Is< float >() == true )
        {
            return new KSComponentIntegral< float >( aComponent, aComponent->As< float >() );
        }

        if( aComponent->Is< double >() == true )
        {
            return new KSComponentIntegral< double >( aComponent, aComponent->As< double >() );
        }

        if( aComponent->Is< KTwoVector >() == true )
        {
            return new KSComponentIntegral< KTwoVector >( aComponent, aComponent->As< KTwoVector >() );
        }

        if( aComponent->Is< KThreeVector >() == true )
        {
            return new KSComponentIntegral< KThreeVector >( aComponent, aComponent->As< KThreeVector >() );
        }

        if( aComponent->Is< KTwoMatrix >() == true )
        {
            return new KSComponentIntegral< KTwoMatrix >( aComponent, aComponent->As< KTwoMatrix >() );
        }

        if( aComponent->Is< KThreeMatrix >() == true )
        {
            return new KSComponentIntegral< KThreeMatrix >( aComponent, aComponent->As< KThreeMatrix >() );
        }

        return NULL;
    }

    typedef KComplexElement< KSComponentIntegralData > KSComponentIntegralBuilder;

    template< >
    inline bool KSComponentIntegralBuilder::Begin()
    {
        fObject = new KSComponentIntegralData;
        return true;
    }

    template< >
    inline bool KSComponentIntegralBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            string tName = aContainer->AsReference< string >();
            fObject->fName = tName;
            return true;
        }
        if( aContainer->GetName() == "group" )
        {
            string tGroupName = aContainer->AsReference< string >();
            fObject->fGroupName = tGroupName;
            return true;
        }
        if( aContainer->GetName() == "component" )
        {
            string tParentName = aContainer->AsReference< string >();
            fObject->fParentName = tParentName;
            return true;
        }
        return false;
    }

    template< >
    inline bool KSComponentIntegralBuilder::End()
    {
        KSComponent* tParentComponent = NULL;
        if( fObject->fGroupName.empty() == false )
        {
            KSComponentGroup* tComponentGroup = KSToolbox::GetInstance()->GetObjectAs< KSComponentGroup >( fObject->fGroupName );
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
                objctmsg( eError ) << "component integral builder could not find component <" << fObject->fParentName << "> in group <" << fObject->fGroupName << ">" << eom;
                return false;
            }
        }
        else
        {
            tParentComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( fObject->fParentName );
        }
        KSComponent* tComponent = BuildOutputIntegral( tParentComponent );
        tComponent->SetName( fObject->fName );
        delete fObject;
        Set( tComponent );
        return true;
    }

}

#endif
