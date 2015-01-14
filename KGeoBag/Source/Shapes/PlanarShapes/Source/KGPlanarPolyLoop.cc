#include "KGPlanarPolyLoop.hh"
#include "KGShapeMessage.hh"

namespace KGeoBag
{

    KGPlanarPolyLoop::KGPlanarPolyLoop() :
            fElements(),
            fLength( 0. ),
            fCentroid( 0., 0. ),
            fAnchor( 0., 0. ),
            fInitialized( false )
    {
    }
    KGPlanarPolyLoop::KGPlanarPolyLoop( const KGPlanarPolyLoop& aCopy ) :
            fElements(),
            fLength( aCopy.fLength ),
            fCentroid( aCopy.fCentroid ),
            fAnchor( aCopy.fAnchor ),
            fInitialized( aCopy.fInitialized )
    {
        const KGPlanarOpenPath* tElement;
        const KGPlanarLineSegment* tLineSegment;
        const KGPlanarArcSegment* tArcSegment;
        for( CIt tIt = aCopy.fElements.begin(); tIt != aCopy.fElements.end(); tIt++ )
        {
            tElement = *tIt;

            tLineSegment = dynamic_cast< const KGPlanarLineSegment* >( tElement );
            if( tLineSegment != NULL )
            {
                fElements.push_back( tLineSegment->Clone() );
                continue;
            }

            tArcSegment = dynamic_cast< const KGPlanarArcSegment* >( tElement );
            if( tArcSegment != NULL )
            {
                fElements.push_back( tArcSegment->Clone() );
                continue;
            }
        }
    }
    KGPlanarPolyLoop::~KGPlanarPolyLoop()
    {
        shapemsg_debug( "destroying a planar poly loop" << eom );

        const KGPlanarOpenPath* tElement;
        for( It tIt = fElements.begin(); tIt != fElements.end(); tIt++ )
        {
            tElement = *tIt;
            delete tElement;
        }
    }

    KGPlanarPolyLoop* KGPlanarPolyLoop::Clone() const
    {
        return new KGPlanarPolyLoop( *this );
    }
    void KGPlanarPolyLoop::CopyFrom( const KGPlanarPolyLoop& aCopy )
    {
        fLength = aCopy.fLength;
        fCentroid = aCopy.fCentroid;
        fAnchor = aCopy.fAnchor;
        fInitialized = aCopy.fInitialized;

        const KGPlanarOpenPath* tElement;
        for( It tIt = fElements.begin(); tIt != fElements.end(); tIt++ )
        {
            tElement = *tIt;
            delete tElement;
        }
        fElements.clear();

        const KGPlanarLineSegment* tLineSegment;
        const KGPlanarArcSegment* tArcSegment;
        for( CIt tIt = aCopy.fElements.begin(); tIt != aCopy.fElements.end(); tIt++ )
        {
            tElement = *tIt;

            tLineSegment = dynamic_cast< const KGPlanarLineSegment* >( tElement );
            if( tLineSegment != NULL )
            {
                fElements.push_back( new KGPlanarLineSegment( *tLineSegment ) );
                continue;
            }

            tArcSegment = dynamic_cast< const KGPlanarArcSegment* >( tElement );
            if( tArcSegment != NULL )
            {
                fElements.push_back( new KGPlanarArcSegment( *tArcSegment ) );
                continue;
            }
        }

        return;
    }

    void KGPlanarPolyLoop::StartPoint( const KTwoVector& aPoint )
    {
        shapemsg_debug( "adding first point to a planar poly line" << eom );
        fInitialized = false;

        const KGPlanarOpenPath* tElement;
        for( It tIt = fElements.begin(); tIt != fElements.end(); tIt++ )
        {
            tElement = *tIt;
            delete tElement;
        }
        fElements.clear();
        fAnchor = aPoint;

        return;
    }
    void KGPlanarPolyLoop::NextLine( const KTwoVector& aVertex, const unsigned int aCount, const double aPower )
    {
        shapemsg_debug( "adding next line to a planar poly line" << eom );
        fInitialized = false;

        if( fElements.empty() == true )
        {
            fElements.push_back( new KGPlanarLineSegment( fAnchor, aVertex, aCount, aPower ) );
        }
        else
        {
            fElements.push_back( new KGPlanarLineSegment( fElements.back()->End(), aVertex, aCount, aPower ) );
        }

        return;
    }
    void KGPlanarPolyLoop::NextArc( const KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong, const unsigned int aCount )
    {
        shapemsg_debug( "adding next arc to a planar poly line" << eom );
        fInitialized = false;

        if( fElements.empty() == true )
        {
            fElements.push_back( new KGPlanarArcSegment( fAnchor, aVertex, aRadius, aLeft, aLong, aCount ) );
        }
        else
        {
            fElements.push_back( new KGPlanarArcSegment( fElements.back()->End(), aVertex, aRadius, aLeft, aLong, aCount ) );
        }

        return;
    }
    void KGPlanarPolyLoop::PreviousLine( const KTwoVector& aVertex, const unsigned int aCount, const double aPower )
    {
        shapemsg_debug( "adding previous line to a planar poly line" << eom );
        fInitialized = false;

        if( fElements.empty() == true )
        {
            fElements.push_back( new KGPlanarLineSegment( aVertex, fAnchor, aCount, aPower ) );
        }
        else
        {
            fElements.push_back( new KGPlanarLineSegment( aVertex, fElements.front()->Start(), aCount, aPower ) );
        }

        return;
    }
    void KGPlanarPolyLoop::PreviousArc( const KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong, const unsigned int aCount )
    {
        shapemsg_debug( "adding previous arc to a planar poly line" << eom );
        fInitialized = false;

        if( fElements.empty() == true )
        {
            fElements.push_back( new KGPlanarArcSegment( aVertex, fAnchor, aRadius, aLeft, aLong, aCount ) );
        }
        else
        {
            fElements.push_back( new KGPlanarArcSegment( aVertex, fElements.front()->Start(), aRadius, aLeft, aLong, aCount ) );
        }

        return;
    }
    void KGPlanarPolyLoop::LastLine( const unsigned int aCount, const double aPower )
    {
        shapemsg_debug( "adding last line to a planar poly loop" << eom );
        fInitialized = false;
        fElements.push_back( new KGPlanarLineSegment( fElements.back()->End(), fElements.front()->Start(), aCount, aPower ) );
    }
    void KGPlanarPolyLoop::LastArc( const double& aRadius, const bool& aLeft, const bool& aLong, const unsigned int aCount )
    {
        shapemsg_debug( "adding last arc to a planar poly loop" << eom );
        fInitialized = false;
        fElements.push_back( new KGPlanarArcSegment( fElements.back()->End(), fElements.front()->Start(), aRadius, aLeft, aLong, aCount ) );
    }

    const KGPlanarPolyLoop::Set& KGPlanarPolyLoop::Elements() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fElements;
    }

    const double& KGPlanarPolyLoop::Length() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fLength;
    }
    const KTwoVector& KGPlanarPolyLoop::Centroid() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fCentroid;
    }
    const KTwoVector& KGPlanarPolyLoop::Anchor() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fAnchor;
    }

    KTwoVector KGPlanarPolyLoop::At( const double& aLength ) const
    {
        if( fInitialized == false )
        {
            Initialize();
        }

        double tLength = aLength;

        if( tLength < 0. )
        {
            return fAnchor;
        }
        if( tLength > fLength )
        {
            return fAnchor;
        }

        for( CIt tIt = fElements.begin(); tIt != fElements.end(); tIt++ )
        {
            if( (*tIt)->Length() > tLength )
            {
                return (*tIt)->At( tLength );
            }
            tLength -= (*tIt)->Length();
        }
        return fAnchor;
    }
    KTwoVector KGPlanarPolyLoop::Point( const KTwoVector& aQuery ) const
    {
        if( fInitialized == false )
        {
            Initialize();
        }

        KTwoVector tCurrent;
        double tCurrentDistanceSquared;

        KTwoVector tNearest;
        double tNearestDistanceSquared;

        CIt tIt = fElements.begin();

        tNearest = (*tIt)->Point( aQuery );
        tNearestDistanceSquared = (tNearest - aQuery).MagnitudeSquared();
        tIt++;

        while( tIt != fElements.end() )
        {
            tCurrent = (*tIt)->Point( aQuery );
            tCurrentDistanceSquared = (tCurrent - aQuery).MagnitudeSquared();
            if( tCurrentDistanceSquared < tNearestDistanceSquared )
            {
                tNearest = tCurrent;
                tNearestDistanceSquared = tCurrentDistanceSquared;
            }
            tIt++;
        }

        return tNearest;
    }
    KTwoVector KGPlanarPolyLoop::Normal( const KTwoVector& aQuery ) const
    {
        if( fInitialized == false )
        {
            Initialize();
        }

        KTwoVector tFirstPoint;
        KTwoVector tFirstNormal;
        double tFirstDistance;

        KTwoVector tSecondPoint;
        KTwoVector tSecondNormal;
        double tSecondDistance;

        KTwoVector tAveragePoint;
        KTwoVector tAverageNormal;
        double tAverageDistance;

        KTwoVector tNearestPoint;
        KTwoVector tNearestNormal;
        double tNearestDistance;

        CIt tIt = fElements.begin();

        tFirstPoint = (*tIt)->Point( aQuery );
        tFirstNormal = (*tIt)->Normal( aQuery );
        tFirstDistance = (aQuery - tFirstPoint).Magnitude();

        tNearestPoint = tFirstPoint;
        tNearestNormal = tFirstNormal;
        tNearestDistance = tFirstDistance;

        tIt++;

        for( ; tIt != fElements.end(); tIt++ )
        {
            tSecondPoint = (*tIt)->Point( aQuery );
            tSecondNormal = (*tIt)->Normal( aQuery );
            tSecondDistance = (aQuery - tSecondPoint).Magnitude();

            tAveragePoint = .5 * (tFirstPoint + tSecondPoint);
            tAverageNormal = (tFirstNormal + tSecondNormal).Unit();
            tAverageDistance = .5 * (tFirstDistance + tSecondDistance);

            if( ((tFirstPoint - tSecondPoint).Magnitude() / (tAveragePoint).Magnitude()) < 1.e-12 )
            {
                if( (fabs( tAverageDistance - tNearestDistance ) / tNearestDistance) < 1.e-12 )
                {
                    tNearestPoint = tAveragePoint;
                    if( tAverageNormal.Dot( aQuery - tAveragePoint ) > 0. )
                    {
                        tNearestNormal = 1. * (aQuery - tAveragePoint).Unit();
                    }
                    else
                    {
                        tNearestNormal = -1. * (aQuery - tAveragePoint).Unit();
                    }
                    tNearestDistance = tAverageDistance;

                    tFirstPoint = tSecondPoint;
                    tFirstNormal = tSecondNormal;
                    tFirstDistance = tSecondDistance;
                    continue;
                }

                if( tAverageDistance < tNearestDistance )
                {
                    tNearestPoint = tAveragePoint;
                    if( tAverageNormal.Dot( aQuery - tAveragePoint ) > 0. )
                    {
                        tNearestNormal = 1. * (aQuery - tAveragePoint).Unit();
                    }
                    else
                    {
                        tNearestNormal = -1. * (aQuery - tAveragePoint).Unit();
                    }
                    tNearestDistance = tAverageDistance;

                    tFirstPoint = tSecondPoint;
                    tFirstNormal = tSecondNormal;
                    tFirstDistance = tSecondDistance;
                    continue;
                }
            }

            if( tSecondDistance < tNearestDistance )
            {
                tNearestPoint = tSecondPoint;
                tNearestNormal = tSecondNormal;
                tNearestDistance = tSecondDistance;

                tFirstPoint = tSecondPoint;
                tFirstNormal = tSecondNormal;
                tFirstDistance = tSecondDistance;
                continue;
            }

            tFirstPoint = tSecondPoint;
            tFirstNormal = tSecondNormal;
            tFirstDistance = tSecondDistance;
        }

        return tNearestNormal;
    }
    bool KGPlanarPolyLoop::Above( const KTwoVector& aQuery ) const
    {
        if( fInitialized == false )
        {
            Initialize();
        }

        KTwoVector tPoint = Point( aQuery );
        KTwoVector tNormal = Normal( aQuery );

        if( tNormal.Dot( aQuery - tPoint ) > 0. )
        {
            return true;
        }

        return false;
    }

    void KGPlanarPolyLoop::Initialize() const
    {
        shapemsg_debug( "initializing a planar poly loop" << eom );

        fAnchor = fElements.front()->Start();

        fLength = 0.;
        fCentroid.X() = 0;
        fCentroid.Y() = 0;

        for( CIt tIt = fElements.begin(); tIt != fElements.end(); tIt++ )
        {
            fLength += (*tIt)->Length();
            fCentroid += (*tIt)->Length() * (*tIt)->Centroid();
        }
        fCentroid /= fLength;

        fInitialized = true;

        return;
    }

}
