#include "KSGeoSide.h"
#include "KSGeoSpace.h"
#include "KSGeometryMessage.h"
#include <limits>

namespace Kassiopeia
{

    KSGeoSide::KSGeoSide() :
            fOutsideParent( NULL ),
            fInsideParent( NULL ),
            fContents()
    {
    }
    KSGeoSide::KSGeoSide( const KSGeoSide& aCopy ) :
            fOutsideParent( NULL ),
            fInsideParent( NULL ),
            fContents( aCopy.fContents )
    {
    }
    KSGeoSide* KSGeoSide::Clone() const
    {
        return new KSGeoSide( *this );
    }
    KSGeoSide::~KSGeoSide()
    {
    }

    void KSGeoSide::On() const
    {
        geomsg_debug( "on geo side <" << this->GetName() << ">" << eom );
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->Activate();
        }
        return;
    }
    void KSGeoSide::Off() const
    {
        geomsg_debug( "off geo side <" << this->GetName() << ">" << eom );
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->Deactivate();
        }
        return;
    }

    KThreeVector KSGeoSide::Point( const KThreeVector& aPoint ) const
    {
        double tDistance;
        KThreeVector tPoint;
        vector< KGSurface* >::const_iterator tSide;

        double tNearestDistance = std::numeric_limits< double >::max();
        KThreeVector tNearestPoint;
        vector< KGSurface* >::const_iterator tNearestSide;

        for( tSide = fContents.begin(); tSide != fContents.end(); tSide++ )
        {
            tPoint = (*tSide)->Point( aPoint );
            tDistance = (tPoint - aPoint).Magnitude();
            if( tDistance < tNearestDistance )
            {
                tNearestDistance = tDistance;
                tNearestPoint = tPoint;
                tNearestSide = tSide;
            }
        }

        return tPoint;
    }
    KThreeVector KSGeoSide::Normal( const KThreeVector& aPoint ) const
    {
        double tDistance;
        KThreeVector tPoint;
        vector< KGSurface* >::const_iterator tSide;

        double tNearestDistance = std::numeric_limits< double >::max();
        KThreeVector tNearestPoint;
        vector< KGSurface* >::const_iterator tNearestSide;

        for( tSide = fContents.begin(); tSide != fContents.end(); tSide++ )
        {
            tPoint = (*tSide)->Point( aPoint );
            tDistance = (tPoint - aPoint).Magnitude();
            if( tDistance < tNearestDistance )
            {
                tNearestDistance = tDistance;
                tNearestPoint = tPoint;
                tNearestSide = tSide;
            }
        }

        return (*tNearestSide)->Normal( aPoint );
    }

    void KSGeoSide::AddContent( KGSurface* aSurface )
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
    void KSGeoSide::RemoveContent( KGSurface* aSurface )
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

    void KSGeoSide::AddCommand( KSCommand* anCommand )
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
    void KSGeoSide::RemoveCommand( KSCommand* anCommand )
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

    void KSGeoSide::InitializeComponent()
    {
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->GetParent()->Initialize();
            (*tCommandIt)->GetChild()->Initialize();
        }

        return;
    }
    void KSGeoSide::DeinitializeComponent()
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
