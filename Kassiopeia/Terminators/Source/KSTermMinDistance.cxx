#include "KSTermMinDistance.h"

#include "KSTerminatorsMessage.h"

#include <limits>

using namespace std;

namespace Kassiopeia
{

    KSTermMinDistance::KSTermMinDistance() :
        fMinDistancePerStep( std::numeric_limits< double >::max() ),
        fMinDistance( 0. ),
        fSurfaces(),
        fSpaces()
    {
    }

    KSTermMinDistance::KSTermMinDistance( const KSTermMinDistance& aCopy ) :
        KSComponent(),
		fMinDistancePerStep( aCopy.fMinDistancePerStep ),
		fMinDistance( aCopy.fMinDistance ),
        fSurfaces( aCopy.fSurfaces ),
        fSpaces( aCopy.fSpaces )
    {
    }

    KSTermMinDistance* KSTermMinDistance::Clone() const
    {
        return new KSTermMinDistance( *this );
    }

    KSTermMinDistance::~KSTermMinDistance()
    {
    }

    void KSTermMinDistance::AddSurface( KGeoBag::KGSurface* aSurface )
    {
        fSurfaces.push_back( aSurface );
        return;
    }

    void KSTermMinDistance::AddSpace( KGeoBag::KGSpace* aSpace )
    {
        fSpaces.push_back( aSpace );
        return;
    }

    void KSTermMinDistance::CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag )
    {
    	fMinDistancePerStep = std::numeric_limits< double >::max();

        KThreeVector tNearestPoint(0., 0., 0.);
        double tMinDist( 0. );

        for( vector< KGeoBag::KGSurface* >::iterator tSurfaceIt = fSurfaces.begin(); tSurfaceIt != fSurfaces.end(); tSurfaceIt++ )
        {
            tNearestPoint = (*tSurfaceIt)->Point( anInitialParticle.GetPosition() );
            tMinDist = (anInitialParticle.GetPosition() - tNearestPoint).Magnitude();
            if( tMinDist < fMinDistancePerStep )
            {
            	fMinDistancePerStep = tMinDist;
            }
        }

        for( vector< KGeoBag::KGSpace* >::iterator tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); tSpaceIt++ )
        {
            tNearestPoint = (*tSpaceIt)->Point( anInitialParticle.GetPosition() );
            tMinDist = (anInitialParticle.GetPosition() - tNearestPoint).Magnitude();
            if( tMinDist < fMinDistancePerStep )
            {
            	fMinDistancePerStep = tMinDist;
            }
        }

        ( fMinDistancePerStep < fMinDistance ) ? aFlag = true : aFlag = false;

        return;
    }
    void KSTermMinDistance::ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const
    {
        aFinalParticle.SetActive( false );
        aFinalParticle.SetLabel( GetName() );
    	return;
    }

    void KSTermMinDistance::InitializeComponent()
    {
        termmsg( eWarning ) << "slower tracking due to activated min distance terminator, better use the navigation for terminating particles at geometries" << eom;
    }

    STATICINT sKSTermMinDistanceDict =
        KSDictionary< KSTermMinDistance >::AddComponent( &KSTermMinDistance::GetMinDistancePerStep, "min_distance" );

}

