#include "KGDiskSurface.hh"

namespace KGeoBag
{

    KGDiskSurface::Visitor::Visitor()
    {
    }
    KGDiskSurface::Visitor::~Visitor()
    {
    }

    KGDiskSurface::KGDiskSurface() :
            KGRotatedPathSurface< KGPlanarLineSegment >(),
            fZ( 0. ),
            fR( 0. ),
            fRadialMeshCount( 8 ),
            fRadialMeshPower( 1. ),
            fAxialMeshCount( 8 )
    {
    }
    KGDiskSurface::~KGDiskSurface()
    {
    }

    void KGDiskSurface::Z( const double& aZ )
    {
        fZ = aZ;
        return;
    }
    void KGDiskSurface::R( const double& anR )
    {
        fR = anR;
        return;
    }
    void KGDiskSurface::RadialMeshCount( const unsigned int& aRadialMeshCount )
    {
        fRadialMeshCount = aRadialMeshCount;
        return;
    }
    void KGDiskSurface::RadialMeshPower( const double& aRadialMeshPower )
    {
        fRadialMeshPower = aRadialMeshPower;
        return;
    }
    void KGDiskSurface::AxialMeshCount( const unsigned int& anAxialMeshCount )
    {
        fAxialMeshCount = anAxialMeshCount;
        return;
    }

    const double& KGDiskSurface::Z() const
    {
        return fZ;
    }
    const double& KGDiskSurface::R() const
    {
        return fR;
    }
    const unsigned int& KGDiskSurface::RadialMeshCount() const
    {
        return fRadialMeshCount;
    }
    const double& KGDiskSurface::RadialMeshPower() const
    {
        return fRadialMeshPower;
    }
    const unsigned int& KGDiskSurface::AxialMeshCount() const
    {
        return fAxialMeshCount;
    }

    void KGDiskSurface::AreaInitialize() const
    {
        fPath->X1( fZ );
        fPath->Y1( 0. );
        fPath->X2( fZ );
        fPath->Y2( fR );

        fPath->MeshCount( fRadialMeshCount );
        fPath->MeshPower( fRadialMeshPower );
        fRotatedMeshCount = fAxialMeshCount;

        KGRotatedLineSegmentSurface::AreaInitialize();
        return;
    }
    void KGDiskSurface::AreaAccept( KGVisitor* aVisitor )
    {
        KGDiskSurface::Visitor* tDiskSurfaceVisitor = dynamic_cast< KGDiskSurface::Visitor* >( aVisitor );
        if( tDiskSurfaceVisitor != 0 )
        {
            tDiskSurfaceVisitor->VisitDiskSurface( this );
            return;
        }
        KGRotatedLineSegmentSurface::AreaAccept( aVisitor );
        return;
    }


}
