#include "KGTorusSpace.hh"

namespace KGeoBag
{

    KGTorusSpace::Visitor::Visitor()
    {
    }
    KGTorusSpace::Visitor::~Visitor()
    {
    }

    KGTorusSpace::KGTorusSpace() :
            fZ( 0. ),
            fR( 0. ),
            fRadius( 0. ),
            fToroidalMeshCount( 64 ),
            fAxialMeshCount( 64 )
    {
    }
    KGTorusSpace::~KGTorusSpace()
    {
    }

    void KGTorusSpace::Z( const double& aZ )
    {
        fZ = aZ;
        return;
    }
    void KGTorusSpace::R( const double& anR )
    {
        fR = anR;
        return;
    }
    void KGTorusSpace::Radius( const double& aRadius )
    {
        fRadius = aRadius;
        return;
    }
    void KGTorusSpace::ToroidalMeshCount( const unsigned int& aToroidalMeshCount )
    {
        fToroidalMeshCount = aToroidalMeshCount;
        return;
    }
    void KGTorusSpace::AxialMeshCount( const unsigned int& anAxialMeshCount )
    {
        fAxialMeshCount = anAxialMeshCount;
        return;
    }

    const double& KGTorusSpace::Z() const
    {
        return fZ;
    }
    const double& KGTorusSpace::R() const
    {
        return fR;
    }
    const double& KGTorusSpace::Radius() const
    {
        return fRadius;
    }
    const unsigned int& KGTorusSpace::ToroidalMeshCount() const
    {
        return fToroidalMeshCount;
    }
    const unsigned int& KGTorusSpace::AxialMeshCount() const
    {
        return fAxialMeshCount;
    }

    void KGTorusSpace::VolumeInitialize( BoundaryContainer& aBoundaryContainer ) const
    {
        fPath->X( fZ );
        fPath->Y( fR );
        fPath->Radius( fRadius );

        fPath->MeshCount( fToroidalMeshCount );
        fRotatedMeshCount = fAxialMeshCount;

        KGRotatedCircleSpace::VolumeInitialize( aBoundaryContainer );
    }
    void KGTorusSpace::VolumeAccept( KGVisitor* aVisitor )
    {
        KGTorusSpace::Visitor* tTorusSpaceVisitor = dynamic_cast< KGTorusSpace::Visitor* >( aVisitor );
        if( tTorusSpaceVisitor != NULL )
        {
            tTorusSpaceVisitor->VisitTorusSpace( this );
            return;
        }
        KGRotatedCircleSpace::VolumeAccept( aVisitor );
        return;
    }

}

