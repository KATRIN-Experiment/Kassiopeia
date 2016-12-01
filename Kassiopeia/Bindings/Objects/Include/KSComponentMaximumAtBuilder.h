#ifndef Kassiopeia_KSComponentMaximumAtBuilder_h_
#define Kassiopeia_KSComponentMaximumAtBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentMaximumAt.h"
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

    class KSComponentMaximumAtData
    {
        public:
            std::string fName;
            std::string fGroupName;
            std::string fParentName;
            std::string fSourceName;
    };

    KSComponent* BuildOutputMaximumAt( KSComponent* aComponent, KSComponent* aSource )
    {

#define BUILD_OUTPUT( xVALUE, xSOURCE )  \
        if(( aComponent->Is< xVALUE >() == true ) && ( aSource->Is< xSOURCE >() == true ))  \
        {                                                                                   \
            return new KSComponentMaximumAt< xVALUE, xSOURCE >(                              \
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
            BUILD_OUTPUT( xVALUE, long )  \
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
        BUILD_OUTPUT_CLASS( unsigned short )
        BUILD_OUTPUT_CLASS( short )
        BUILD_OUTPUT_CLASS( unsigned int )
        BUILD_OUTPUT_CLASS( int )
        BUILD_OUTPUT_CLASS( unsigned long )
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

    typedef KComplexElement< KSComponentMaximumAtData > KSComponentMaximumAtBuilder;

    template< >
    inline bool KSComponentMaximumAtBuilder::Begin()
    {
        fObject = new KSComponentMaximumAtData;
        return true;
    }

    template< >
    inline bool KSComponentMaximumAtBuilder::AddAttribute( KContainer* aContainer )
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
            objctmsg( eWarning ) <<"deprecated warning in KSComponentMaximumAtBuilder: Please use the attribute <parent> instead <component>"<<eom;
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
        if( aContainer->GetName() == "source" )
        {
            std::string tSourceName = aContainer->AsReference< std::string >();
            fObject->fSourceName = tSourceName;
            return true;
        }
        return false;
    }

    template< >
    inline bool KSComponentMaximumAtBuilder::End()
    {
        KSComponent* tParentComponent = NULL;
        KSComponent* tSourceComponent = NULL;
        if( fObject->fGroupName.empty() == false )
        {
            KSComponentGroup* tComponentGroup = KToolbox::GetInstance().Get< KSComponentGroup >( fObject->fGroupName );
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
                objctmsg( eError ) << "component maximum_at builder could not find component <" << fObject->fParentName << "> in group <" << fObject->fGroupName << ">" << eom;
                return false;
            }
            if( tSourceComponent == NULL )
            {
                objctmsg( eError ) << "component maximum_at builder could not find component <" << fObject->fSourceName << "> in group <" << fObject->fGroupName << ">" << eom;
                return false;
            }
        }
        else
        {
            tParentComponent = KToolbox::GetInstance().Get< KSComponent >( fObject->fParentName );
            tSourceComponent = KToolbox::GetInstance().Get< KSComponent >( fObject->fSourceName );
        }
        KSComponent* tComponent = BuildOutputMaximumAt( tParentComponent, tSourceComponent );
        tComponent->SetName( fObject->fName );
        delete fObject;
        Set( tComponent );
        return true;
    }

}

#endif
