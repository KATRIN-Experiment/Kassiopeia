#ifndef KGELECTROMAGNETBUILDER_HH_
#define KGELECTROMAGNETBUILDER_HH_

#include "KGElectromagnet.hh"
#include "KField.h"

namespace KGeoBag
{

    class KGElectromagnetAttributor :
        public KTagged,
        public KGElectromagnetData
    {
        public:
            KGElectromagnetAttributor();
            virtual ~KGElectromagnetAttributor();

        public:
            void AddSurface( KGSurface* aSurface );
            void AddSpace( KGSpace* aSpace );

        private:
            std::vector< KGSurface* > fSurfaces;
            std::vector< KGSpace* > fSpaces;
            K_SET_GET( double, LineCurrent )
            K_SET_GET( double, ScalingFactor )
            K_SET_GET( double, Direction )
    };

}

#include "KComplexElement.hh"

namespace katrin
{

    typedef KComplexElement< KGeoBag::KGElectromagnetAttributor > KGElectromagnetBuilder;

    template< >
    inline bool KGElectromagnetBuilder::AddAttribute( KContainer* aContainer )
    {
        using namespace KGeoBag;
        using namespace std;

        if( aContainer->GetName() == "name" )
        {
            fObject->SetName( aContainer->AsReference< string >() );
            return true;
        }
        if( aContainer->GetName() == "current" )
        {
            fObject->SetLineCurrent( aContainer->AsReference< double >() );
            return true;
        }
        if( aContainer->GetName() == "scaling_factor" )
        {
            fObject->SetScalingFactor( aContainer->AsReference< double >() );
            return true;
        }
        if( aContainer->GetName() == "direction" )
        {
        	string tDirection = aContainer->AsReference<string>();
        	if ( tDirection == string("clockwise") || tDirection == string("normal") )
        	{
				fObject->SetDirection( 1.0 );
				return true;
        	}
        	if ( tDirection == string("counter_clockwise") || tDirection == string("reversed"))
        	{
				fObject->SetDirection( -1.0 );
				return true;
        	}
            coremsg( eWarning ) << "Dont know the direction <"<< tDirection<<">" << eom;
            return false;
        }
        if( aContainer->GetName() == "surfaces" )
        {
            vector< KGSurface* > tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces( aContainer->AsReference< string >() );
            vector< KGSurface* >::const_iterator tSurfaceIt;
            KGSurface* tSurface;

            if( tSurfaces.size() == 0 )
            {
                coremsg( eWarning ) << "no surfaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return false;
            }

            for( tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++ )
            {
                tSurface = *tSurfaceIt;
                fObject->AddSurface( tSurface );
            }
            return true;
        }
        if( aContainer->GetName() == "spaces" )
        {
            vector< KGSpace* > tSpaces = KGInterface::GetInstance()->RetrieveSpaces( aContainer->AsReference< string >() );
            vector< KGSpace* >::const_iterator tSpaceIt;
            KGSpace* tSpace;

            if( tSpaces.size() == 0 )
            {
                coremsg( eWarning ) << "no spaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return false;
            }

            for( tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++ )
            {
                tSpace = *tSpaceIt;
                fObject->AddSpace( tSpace );
            }
            return true;
        }
        return false;
    }

    template< >
    inline bool KGElectromagnetBuilder::End()
    {
    	fObject->SetCurrent( fObject->GetLineCurrent() * fObject->GetScalingFactor() * fObject->GetDirection() );
    	return true;
    }

}

#endif
