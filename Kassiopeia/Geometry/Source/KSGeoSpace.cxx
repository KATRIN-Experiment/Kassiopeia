#include "KSGeoSpace.h"
#include "KSGeoSurface.h"
#include "KSGeoSide.h"
#include "KSGeometryMessage.h"
#include <limits>

namespace Kassiopeia
{

    KSGeoSpace::KSGeoSpace() :
            fContents(),
            fCommands()
    {
    }
    KSGeoSpace::KSGeoSpace( const KSGeoSpace& aCopy ) :
            fContents( aCopy.fContents ),
            fCommands( aCopy.fCommands )
    {
    }
    KSGeoSpace* KSGeoSpace::Clone() const
    {
        return new KSGeoSpace( *this );
    }
    KSGeoSpace::~KSGeoSpace()
    {
    }

    void KSGeoSpace::Enter() const
    {
        geomsg_debug( "enter geo space <" << this->GetName() << ">" << eom )
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->Activate();
        }
        return;
    }
    void KSGeoSpace::Exit() const
    {
        geomsg_debug( "exit geo space <" << this->GetName() << ">" << eom )
        vector< KSCommand* >::reverse_iterator tCommandIt;
        for( tCommandIt = fCommands.rbegin(); tCommandIt != fCommands.rend(); tCommandIt++ )
        {
            (*tCommandIt)->Deactivate();
        }
        return;
    }

    bool KSGeoSpace::Outside( const KThreeVector& aPoint ) const
    {
        bool tOutside;
        vector< KGSpace* >::const_iterator tSpace;

        for( tSpace = fContents.begin(); tSpace != fContents.end(); tSpace++ )
        {
            tOutside = (*tSpace)->Outside( aPoint );
            if( tOutside == true )
            {
                return true;
            }
        }

        return false;
    }

    KThreeVector KSGeoSpace::Point( const KThreeVector& aPoint ) const
    {
        double tDistance;
        KThreeVector tPoint;
        vector< KGSpace* >::const_iterator tSpace;
        double tNearestDistance = std::numeric_limits< double >::max();
        KThreeVector tNearestPoint;
        vector< KGSpace* >::const_iterator tNearestSpace;

        for( tSpace = fContents.begin(); tSpace != fContents.end(); tSpace++ )
        {
            tPoint = (*tSpace)->Point( aPoint );
            tDistance = (tPoint - aPoint).Magnitude();
            if( tDistance < tNearestDistance )
            {
                tNearestDistance = tDistance;
                tNearestPoint = tPoint;
                tNearestSpace = tSpace;
            }
        }

        return tPoint;
    }
    KThreeVector KSGeoSpace::Normal( const KThreeVector& aPoint ) const
    {
        double tDistance;
        KThreeVector tPoint;
        vector< KGSpace* >::const_iterator tSpace;

        double tNearestDistance = std::numeric_limits< double >::max();
        KThreeVector tNearestPoint;
        vector< KGSpace* >::const_iterator tNearestSpace;

        for( tSpace = fContents.begin(); tSpace != fContents.end(); tSpace++ )
        {
            tPoint = (*tSpace)->Point( aPoint );
            tDistance = (tPoint - aPoint).Magnitude();
            if( tDistance < tNearestDistance )
            {
                tNearestDistance = tDistance;
                tNearestPoint = tPoint;
                tNearestSpace = tSpace;
            }
        }

        return (*tNearestSpace)->Normal( aPoint );
    }

    void KSGeoSpace::AddContent( KGSpace* aSpace )
    {
        vector< KGSpace* >::iterator tSpace;
        for( tSpace = fContents.begin(); tSpace != fContents.end(); tSpace++ )
        {
            if( (*tSpace) == aSpace )
            {
                //todo: warning here
                return;
            }
        }
        fContents.push_back( aSpace );
        return;
    }
    void KSGeoSpace::RemoveContent( KGSpace* aSpace )
    {
        vector< KGSpace* >::iterator tSpace;
        for( tSpace = fContents.begin(); tSpace != fContents.end(); tSpace++ )
        {
            if( (*tSpace) == aSpace )
            {
                fContents.erase( tSpace );
                return;
            }
        }
        //todo: warning here
        return;
    }

    void KSGeoSpace::AddCommand( KSCommand* anCommand )
    {
        vector< KSCommand* >::iterator tCommand;
        for( tCommand = fCommands.begin(); tCommand != fCommands.end(); tCommand++ )
        {
            if( (*tCommand) == anCommand )
            {
                //todo: warning here
                return;
            }
        }
        fCommands.push_back( anCommand );
        return;
    }
    void KSGeoSpace::RemoveCommand( KSCommand* anCommand )
    {
        vector< KSCommand* >::iterator tCommand;
        for( tCommand = fCommands.begin(); tCommand != fCommands.end(); tCommand++ )
        {
            if( (*tCommand) == anCommand )
            {
                fCommands.erase( tCommand );
                return;
            }
        }
        //todo: warning here
        return;
    }

    void KSGeoSpace::InitializeComponent()
    {
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->GetParent()->Initialize();
            (*tCommandIt)->GetChild()->Initialize();
        }

        vector< KSSpace* >::iterator tGeoSpaceIt;
        for( tGeoSpaceIt = fSpaces.begin(); tGeoSpaceIt != fSpaces.end(); tGeoSpaceIt++ )
        {
            (*tGeoSpaceIt)->Initialize();
        }
        vector< KSSurface* >::iterator tGeoSurfaceIt;
        for( tGeoSurfaceIt = fSurfaces.begin(); tGeoSurfaceIt != fSurfaces.end(); tGeoSurfaceIt++ )
        {
            (*tGeoSurfaceIt)->Initialize();
        }
        vector< KSSide* >::iterator tGeoSideIt;
        for( tGeoSideIt = fSides.begin(); tGeoSideIt != fSides.end(); tGeoSideIt++ )
        {
            (*tGeoSideIt)->Initialize();
        }

        return;
    }
    void KSGeoSpace::DeinitializeComponent()
    {
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->GetParent()->Deinitialize();
            (*tCommandIt)->GetChild()->Deinitialize();
        }

        vector< KSSpace* >::iterator tGeoSpaceIt;
        for( tGeoSpaceIt = fSpaces.begin(); tGeoSpaceIt != fSpaces.end(); tGeoSpaceIt++ )
        {
            (*tGeoSpaceIt)->Deinitialize();
        }
        vector< KSSurface* >::iterator tGeoSurfaceIt;
        for( tGeoSurfaceIt = fSurfaces.begin(); tGeoSurfaceIt != fSurfaces.end(); tGeoSurfaceIt++ )
        {
            (*tGeoSurfaceIt)->Deinitialize();
        }
        vector< KSSide* >::iterator tGeoSideIt;
        for( tGeoSideIt = fSides.begin(); tGeoSideIt != fSides.end(); tGeoSideIt++ )
        {
            (*tGeoSideIt)->Deinitialize();
        }

        return;
    }

}
