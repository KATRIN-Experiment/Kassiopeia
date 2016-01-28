#include "KGCutConeSurface.hh"

namespace KGeoBag
{

    KGCutConeSurface::Visitor::Visitor()
    {
    }
    KGCutConeSurface::Visitor::~Visitor()
    {
    }

    KGCutConeSurface::KGCutConeSurface() :
            fZ1( 0. ),
            fR1( 0. ),
            fZ2( 0. ),
            fR2( 0. ),
            fLongitudinalMeshCount( 8 ),
            fLongitudinalMeshPower( 1. ),
            fAxialMeshCount( 8 )
    {
    }
    KGCutConeSurface::~KGCutConeSurface()
    {
    }

    void KGCutConeSurface::Z1( const double& aZ1 )
    {
        fZ1 = aZ1;
        return;
    }
    void KGCutConeSurface::R1( const double& anR1 )
    {
        fR1 = anR1;
        return;
    }
    void KGCutConeSurface::Z2( const double& aZ2 )
    {
        fZ2 = aZ2;
        return;
    }
    void KGCutConeSurface::R2( const double& anR2 )
    {
        fR2 = anR2;
        return;
    }
    void KGCutConeSurface::LongitudinalMeshCount( const unsigned int& aLongitudinalMeshCount )
    {
        fLongitudinalMeshCount = aLongitudinalMeshCount;
        return;
    }
    void KGCutConeSurface::LongitudinalMeshPower( const double& aLongitudinalMeshPower )
    {
        fLongitudinalMeshPower = aLongitudinalMeshPower;
        return;
    }
    void KGCutConeSurface::AxialMeshCount( const unsigned int& anAxialMeshCount )
    {
        fAxialMeshCount = anAxialMeshCount;
        return;
    }

    const double& KGCutConeSurface::Z1() const
    {
        return fZ1;
    }
    const double& KGCutConeSurface::R1() const
    {
        return fR1;
    }
    const double& KGCutConeSurface::Z2() const
    {
        return fZ2;
    }
    const double& KGCutConeSurface::R2() const
    {
        return fR2;
    }
    const unsigned int& KGCutConeSurface::LongitudinalMeshCount() const
    {
        return fLongitudinalMeshCount;
    }
    const double& KGCutConeSurface::LongitudinalMeshPower() const
    {
        return fLongitudinalMeshPower;
    }
    const unsigned int& KGCutConeSurface::AxialMeshCount() const
    {
        return fAxialMeshCount;
    }

    void KGCutConeSurface::AreaInitialize() const
    {
        if( fZ1 < fZ2 )
        {
            fPath->X1( fZ2 );
            fPath->Y1( fR2 );
            fPath->X2( fZ1 );
            fPath->Y2( fR1 );
        }
        else
        {
            fPath->X1( fZ1 );
            fPath->Y1( fR1 );
            fPath->X2( fZ2 );
            fPath->Y2( fR2 );
        }

        fPath->MeshCount( fLongitudinalMeshCount );
        fPath->MeshPower( fLongitudinalMeshPower );
        fRotatedMeshCount = fAxialMeshCount;

        KGRotatedLineSegmentSurface::AreaInitialize();
        return;
    }
    void KGCutConeSurface::AreaAccept( KGVisitor* aVisitor )
    {
        KGCutConeSurface::Visitor* tCutConeSurfaceVisitor = dynamic_cast< KGCutConeSurface::Visitor* >( aVisitor );
        if( tCutConeSurfaceVisitor != NULL )
        {
            tCutConeSurfaceVisitor->VisitCutConeSurface( this );
            return;
        }
        KGRotatedLineSegmentSurface::AreaAccept( aVisitor );
        return;
    }

}
