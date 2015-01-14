#include "KSGeoSurface.h"
#include "KSGeoSpace.h"
#include "KSGeometryMessage.h"
#include <limits>

namespace Kassiopeia
{

    KSGeoSurface::KSGeoSurface() :
            fParent( NULL ),
            fContents()
    {
    }
    KSGeoSurface::KSGeoSurface( const KSGeoSurface& aCopy ) :
            fParent( NULL ),
            fContents( aCopy.fContents )
    {
    }
    KSGeoSurface* KSGeoSurface::Clone() const
    {
        return new KSGeoSurface( *this );
    }
    KSGeoSurface::~KSGeoSurface()
    {
    }

    void KSGeoSurface::On() const
    {
        geomsg_debug( "on geo surface <" << this->GetName() << ">" << eom )
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->Activate();
        }
        return;
    }
    void KSGeoSurface::Off() const
    {
        geomsg_debug( "off geo surface <" << this->GetName() << ">" << eom )
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->Deactivate();
        }
        return;
    }

    KThreeVector KSGeoSurface::Point( const KThreeVector& aPoint ) const
    {
        double tDistance;
        KThreeVector tPoint;
        vector< KGSurface* >::const_iterator tSurface;

        double tNearestDistance = std::numeric_limits< double >::max();
        KThreeVector tNearestPoint;
        vector< KGSurface* >::const_iterator tNearestSurface;

        for( tSurface = fContents.begin(); tSurface != fContents.end(); tSurface++ )
        {
            tPoint = (*tSurface)->Point( aPoint );
            tDistance = (tPoint - aPoint).Magnitude();
            if( tDistance < tNearestDistance )
            {
                tNearestDistance = tDistance;
                tNearestPoint = tPoint;
                tNearestSurface = tSurface;
            }
        }

        return tPoint;
    }
    KThreeVector KSGeoSurface::Normal( const KThreeVector& aPoint ) const
    {
        double tDistance;
        KThreeVector tPoint;
        vector< KGSurface* >::const_iterator tSurface;

        double tNearestDistance = std::numeric_limits< double >::max();
        KThreeVector tNearestPoint;
        vector< KGSurface* >::const_iterator tNearestSurface;

        for( tSurface = fContents.begin(); tSurface != fContents.end(); tSurface++ )
        {
            tPoint = (*tSurface)->Point( aPoint );
            tDistance = (tPoint - aPoint).Magnitude();
            if( tDistance < tNearestDistance )
            {
                tNearestDistance = tDistance;
                tNearestPoint = tPoint;
                tNearestSurface = tSurface;
            }
        }

        return (*tNearestSurface)->Normal( aPoint );
    }

    void KSGeoSurface::AddContent( KGSurface* aSurface )
    {
        vector< KGSurface* >::iterator tSurface;
        for( tSurface = fContents.begin(); tSurface != fContents.end(); tSurface++ )
        {
            if( (*tSurface) == aSurface )
            {
                //todo: warning here
                return;
            }
        }
        fContents.push_back( aSurface );
        return;
    }
    void KSGeoSurface::RemoveContent( KGSurface* aSurface )
    {
        vector< KGSurface* >::iterator tSurface;
        for( tSurface = fContents.begin(); tSurface != fContents.end(); tSurface++ )
        {
            if( (*tSurface) == aSurface )
            {
                fContents.erase( tSurface );
                return;
            }
        }
        //todo: warning here
        return;
    }

    void KSGeoSurface::AddCommand( KSCommand* anCommand )
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
    void KSGeoSurface::RemoveCommand( KSCommand* anCommand )
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

    void KSGeoSurface::InitializeComponent()
    {
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->GetParent()->Initialize();
            (*tCommandIt)->GetChild()->Initialize();
        }

        return;
    }
    void KSGeoSurface::DeinitializeComponent()
    {
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->GetParent()->Deinitialize();
            (*tCommandIt)->GetChild()->Deinitialize();
        }

        return;
    }

}
