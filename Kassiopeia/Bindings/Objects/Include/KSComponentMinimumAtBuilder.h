#ifndef Kassiopeia_KSComponentMinimumAtBuilder_h_
#define Kassiopeia_KSComponentMinimumAtBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentMinimumAt.h"
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

    class KSComponentMinimumAtData
    {
        public:
            string fName;
            string fGroupName;
            string fParentName;
            string fSourceName;
    };

    KSComponent* BuildOutputMinimumAt( KSComponent* aComponent, KSComponent* aSource )
    {

#define BUILD_OUTPUT( xVALUE, xSOURCE )  \
        if(( aComponent->Is< xVALUE >() == true ) && ( aSource->Is< xSOURCE >() == true ))  \
        {                                                                                   \
            return new KSComponentMinimumAt< xVALUE, xSOURCE >(                              \
                    aComponent, aComponent->As< xVALUE >(), aSource->As< xSOURCE >() );     \
        }

#define BUILD_OUTPUT_CLASS( xVALUE )  \
        if( aComponent->Is< xVALUE >() == true )   \
        {                                          \
            BUILD_OUTPUT( xVALUE, bool )           \
            BUILD_OUTPUT( xVALUE, unsigned char )  \
            BUILD_OUTPUT( xVALUE, char )           \
            BUILD_OUTPUT( xVALUE, unsigned short ) \
            BUILD_OUTPUT( xVALUE, short )          \
            BUILD_OUTPUT( xVALUE, unsigned int )   \
            BUILD_OUTPUT( xVALUE, int )            \
            BUILD_OUTPUT( xVALUE, unsigned long )  \
            BUILD_OUTPUT( xVALUE, float )          \
            BUILD_OUTPUT( xVALUE, double )         \
            BUILD_OUTPUT( xVALUE, KTwoVector )     \
            BUILD_OUTPUT( xVALUE, KThreeVector )   \
            BUILD_OUTPUT( xVALUE, KTwoMatrix )     \
            BUILD_OUTPUT( xVALUE, KThreeMatrix )   \
        }

        BUILD_OUTPUT_CLASS( bool )
        BUILD_OUTPUT_CLASS( unsigned char )
        BUILD_OUTPUT_CLASS( char )
        BUILD_OUTPUT_CLASS( short )
        BUILD_OUTPUT_CLASS( short )
        BUILD_OUTPUT_CLASS( int )
        BUILD_OUTPUT_CLASS( int )
        BUILD_OUTPUT_CLASS( long )
        BUILD_OUTPUT_CLASS( long )
        BUILD_OUTPUT_CLASS( float )
        BUILD_OUTPUT_CLASS( double )
        BUILD_OUTPUT_CLASS( KTwoVector )
        BUILD_OUTPUT_CLASS( KThreeVector )
        BUILD_OUTPUT_CLASS( KTwoMatrix )
        BUILD_OUTPUT_CLASS( KThreeMatrix )

#undef BUILD_OUTPUT_CLASS
#undef BUILD_OUTPUT

        return NULL;
    }

    typedef KComplexElement< KSComponentMinimumAtData > KSComponentMinimumAtBuilder;

    template< >
    inline bool KSComponentMinimumAtBuilder::Begin()
    {
        fObject = new KSComponentMinimumAtData;
        return true;
    }

    template< >
    inline bool KSComponentMinimumAtBuilder::AddAttribute( KContainer* aContainer )
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
            objctmsg( eWarning ) <<"deprecated warning in KSComponentMinimumAtBuilder: Please use the attribute <parent> instead <component>"<<eom;
            string tParentName = aContainer->AsReference< string >();
            fObject->fParentName = tParentName;
            return true;
        }
        if(  aContainer->GetName() == "parent" )
        {
            string tParentName = aContainer->AsReference< string >();
            fObject->fParentName = tParentName;
            return true;
        }
        if( aContainer->GetName() == "source" )
        {
            string tSourceName = aContainer->AsReference< string >();
            fObject->fSourceName = tSourceName;
            return true;
        }
        return false;
    }

    template< >
    inline bool KSComponentMinimumAtBuilder::End()
    {
        KSComponent* tParentComponent = NULL;
        KSComponent* tSourceComponent = NULL;
        if( fObject->fGroupName.empty() == false )
        {
            KSComponentGroup* tComponentGroup = KSToolbox::GetInstance()->GetObjectAs< KSComponentGroup >( fObject->fGroupName );
            for( unsigned int tIndex = 0; tIndex < tComponentGroup->ComponentCount(); tIndex++ )
            {
                KSComponent* tGroupComponent = tComponentGroup->ComponentAt( tIndex );
                if( tGroupComponent->GetName() == fObject->fParentName )
                {
                    tParentComponent = tGroupComponent;
                }
                if( tGroupComponent->GetName() == fObject->fSourceName )
                {
                    tSourceComponent = tGroupComponent;
                }
            }
            if( tParentComponent == NULL )
            {
                objctmsg( eError ) << "component minimum_at builder could not find component <" << fObject->fParentName << "> in group <" << fObject->fGroupName << ">" << eom;
                return false;
            }
            if( tSourceComponent == NULL )
            {
                objctmsg( eError ) << "component minimum_at builder could not find component <" << fObject->fSourceName << "> in group <" << fObject->fGroupName << ">" << eom;
                return false;
            }
        }
        else
        {
            tParentComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( fObject->fParentName );
            tSourceComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( fObject->fSourceName );
        }
        KSComponent* tComponent = BuildOutputMinimumAt( tParentComponent, tSourceComponent );
        tComponent->SetName( fObject->fName );
        delete fObject;
        Set( tComponent );
        return true;
    }

}

#endif
