#include "KGPlanarLineSegment.hh"
#include "KGShapeMessage.hh"

namespace KGeoBag
{

    KGPlanarLineSegment::KGPlanarLineSegment() :
            KGPlanarOpenPath(),
            fStart( 0., 0. ),
            fEnd( 0., 0. ),
            fMeshCount( 1 ),
            fMeshPower( 1. ),
            fLength( 0. ),
            fCentroid( 0., 0. ),
            fXUnit( 1., 0. ),
            fYUnit( 0., 1. ),
            fInitialized( false )
    {
    }
    KGPlanarLineSegment::KGPlanarLineSegment( const KGPlanarLineSegment& aCopy ) :
            KGPlanarOpenPath(),
            fStart( aCopy.fStart ),
            fEnd( aCopy.fEnd ),
            fMeshCount( aCopy.fMeshCount ),
            fMeshPower( aCopy.fMeshPower ),
            fLength( aCopy.fLength ),
            fCentroid( aCopy.fCentroid ),
            fXUnit( aCopy.fXUnit ),
            fYUnit( aCopy.fYUnit ),
            fInitialized( aCopy.fInitialized )
    {
    }
    KGPlanarLineSegment::KGPlanarLineSegment( const double& anX1, const double& aY1, const double& anX2, const double& aY2, const unsigned int aCount, const double aPower ) :
            KGPlanarOpenPath(),
            fStart( anX1, aY1 ),
            fEnd( anX2, aY2 ),
            fMeshCount( aCount ),
            fMeshPower( aPower ),
            fLength( 0. ),
            fCentroid( 0., 0. ),
            fXUnit( 1., 0. ),
            fYUnit( 0., 1. ),
            fInitialized( false )
    {
    }
    KGPlanarLineSegment::KGPlanarLineSegment( const KTwoVector& aStart, const KTwoVector& anEnd, const unsigned int aCount, const double aPower ) :
            KGPlanarOpenPath(),
            fStart( aStart ),
            fEnd( anEnd ),
            fMeshCount( aCount ),
            fMeshPower( aPower ),
            fLength( 0. ),
            fCentroid( 0., 0. ),
            fXUnit( 1., 0. ),
            fYUnit( 0., 1. ),
            fInitialized( false )
    {
    }
    KGPlanarLineSegment::~KGPlanarLineSegment()
    {
        shapemsg_debug( "destroying a planar line segment" << eom );
    }

    KGPlanarLineSegment* KGPlanarLineSegment::Clone() const
    {
        return new KGPlanarLineSegment( *this );
    }
    void KGPlanarLineSegment::CopyFrom( const KGPlanarLineSegment& aCopy )
    {
        fStart = aCopy.fStart;
        fEnd = aCopy.fEnd;
        fMeshCount = aCopy.fMeshCount;
        fMeshPower = aCopy.fMeshPower;
        fLength = aCopy.fLength;
        fCentroid = aCopy.fCentroid;
        fXUnit = aCopy.fXUnit;
        fYUnit = aCopy.fYUnit;
        fInitialized = aCopy.fInitialized;
        return;
    }

    void KGPlanarLineSegment::Start( const KTwoVector& aStart )
    {
        fInitialized = false;
        fStart = aStart;
        return;
    }
    void KGPlanarLineSegment::X1( const double& aValue )
    {
        fInitialized = false;
        fStart.X() = aValue;
        return;
    }
    void KGPlanarLineSegment::Y1( const double& aValue )
    {
        fInitialized = false;
        fStart.Y() = aValue;
        return;
    }
    void KGPlanarLineSegment::End( const KTwoVector& aEnd )
    {
        fInitialized = false;
        fEnd = aEnd;
        return;
    }
    void KGPlanarLineSegment::X2( const double& aValue )
    {
        fInitialized = false;
        fEnd.X() = aValue;
        return;
    }
    void KGPlanarLineSegment::Y2( const double& aValue )
    {
        fInitialized = false;
        fEnd.Y() = aValue;
        return;
    }
    void KGPlanarLineSegment::MeshCount( const unsigned int& aCount )
    {
        fInitialized = false;
        fMeshCount = aCount;
        return;
    }
    void KGPlanarLineSegment::MeshPower( const double& aPower )
    {
        fInitialized = false;
        fMeshPower = aPower;
        return;
    }

    const KTwoVector& KGPlanarLineSegment::Start() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fStart;
    }
    const double& KGPlanarLineSegment::X1() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fStart.X();
    }
    const double& KGPlanarLineSegment::Y1() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fStart.Y();
    }
    const KTwoVector& KGPlanarLineSegment::End() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fEnd;
    }
    const double& KGPlanarLineSegment::X2() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fEnd.X();
    }
    const double& KGPlanarLineSegment::Y2() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fEnd.Y();
    }
    const unsigned int& KGPlanarLineSegment::MeshCount() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fMeshCount;
    }
    const double& KGPlanarLineSegment::MeshPower() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fMeshPower;
    }

    const double& KGPlanarLineSegment::Length() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fLength;
    }
    const KTwoVector& KGPlanarLineSegment::Centroid() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fCentroid;
    }
    const KTwoVector& KGPlanarLineSegment::XUnit() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fXUnit;
    }
    const KTwoVector& KGPlanarLineSegment::YUnit() const
    {
        if( fInitialized == false )
        {
            Initialize();
        }
        return fYUnit;
    }

    KTwoVector KGPlanarLineSegment::At( const double& aLength ) const
    {
        if( fInitialized == false )
        {
            Initialize();
        }

        double tLength = aLength / fLength;
        return fStart + tLength * (fEnd - fStart);
    }

    KTwoVector KGPlanarLineSegment::Point( const KTwoVector& aQuery ) const
    {
        if( fInitialized == false )
        {
            Initialize();
        }

        double tY = fYUnit.X() * (aQuery.X() - fStart.X()) + fYUnit.Y() * (aQuery.Y() - fStart.Y());

        if( tY < 0. )
        {
            tY = 0.;
        }
        if( tY > fLength )
        {
            tY = fLength;
        }

        return fStart + tY * fYUnit;
    }
    KTwoVector KGPlanarLineSegment::Normal( const KTwoVector& /*aQuery*/) const
    {
        if( fInitialized == false )
        {
            Initialize();
        }

        return fXUnit;

    }
    bool KGPlanarLineSegment::Above( const KTwoVector& aQuery ) const
    {
        if( fInitialized == false )
        {
            Initialize();
        }

        double tX = fXUnit.X() * (aQuery.X() - fStart.X()) + fXUnit.Y() * (aQuery.Y() - fStart.Y());

        if( tX > 0. )
        {
            return true;
        }
        return false;
    }

    void KGPlanarLineSegment::Initialize() const
    {
        shapemsg_debug( "initializing a planar line segment" << eom );

        fLength = (fEnd - fStart).Magnitude();
        fCentroid = (1. / 2.) * (fStart + fEnd);
        fXUnit = (fEnd - fStart).Orthogonal( false ).Unit();
        fYUnit = (fEnd - fStart).Unit();

        fInitialized = true;

        return;
    }

}
