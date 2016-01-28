#ifndef Kassiopeia_KSGeoSideBuilder_h_
#define Kassiopeia_KSGeoSideBuilder_h_

#include "KComplexElement.hh"
#include "KSGeoSide.h"
#include "KSToolbox.h"
#include "KSOperatorsMessage.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGeoSide > KSGeoSideBuilder;

    template< >
    inline bool KSGeoSideBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSGeoSide::SetName );
            return true;
        }
        if( aContainer->GetName() == "surfaces" )
        {
            vector< KGSurface* > tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces( aContainer->AsReference< string >() );
            vector< KGSurface* >::const_iterator tSurfaceIt;
            KGSurface* tSurface;

            if( tSurfaces.size() == 0 )
            {
                oprmsg( eWarning ) << "no surfaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return false;
            }

            for( tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++ )
            {
                tSurface = *tSurfaceIt;
                fObject->AddContent( tSurface );
            }
            return true;
        }
        if( aContainer->GetName() == "spaces" )
        {
            vector< KGSpace* > tSpaces = KGInterface::GetInstance()->RetrieveSpaces( aContainer->AsReference< string >() );
            vector< KGSpace* >::const_iterator tSpaceIt;
            KGSpace* tSpace;
            const vector< KGSurface* >* tSurfaces;
            vector< KGSurface* >::const_iterator tSurfaceIt;
            KGSurface* tSurface;

            if( tSpaces.size() == 0 )
            {
                oprmsg( eWarning ) << "no spaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return false;
            }

            for( tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++ )
            {
                tSpace = *tSpaceIt;
                tSurfaces = tSpace->GetBoundaries();
                for( tSurfaceIt = tSurfaces->begin(); tSurfaceIt != tSurfaces->end(); tSurfaceIt++ )
                {
                    tSurface = *tSurfaceIt;
                    fObject->AddContent( tSurface );
                }
            }
            return true;
        }
        return false;
    }

    template< >
    inline bool KSGeoSideBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSCommand >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSGeoSide::AddCommand );
            return true;
        }
        return false;
    }

}

#endif
