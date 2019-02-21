#ifndef KGROTATEDCLOSEDPATHSPACE_HH_
#define KGROTATEDCLOSEDPATHSPACE_HH_

#include "KGVolume.hh"
#include "KGPlanarClosedPath.hh"
#include "KGRotatedPathSurface.hh"
#include "KGShapeMessage.hh"

namespace KGeoBag
{

    template< class XPathType >
    class KGRotatedClosedPathSpace :
        public KGVolume
    {
        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitRotatedClosedPathSpace( KGRotatedClosedPathSpace* aRotatedClosedPathSpace ) = 0;
            };

        public:
            KGRotatedClosedPathSpace() :
                    KGVolume(),
                    fPath( new XPathType() ),
                    fRotatedMeshCount( 64 )
            {
                CompilerCheck();
            }
            KGRotatedClosedPathSpace( const KGRotatedClosedPathSpace< XPathType >& aCopy ) :
                    KGVolume( aCopy ),
                    fPath( aCopy.fPath ),
                    fRotatedMeshCount( aCopy.fRotatedMeshCount )
            {
            }
            KGRotatedClosedPathSpace( const std::shared_ptr< XPathType >& aPath ) :
                    KGVolume(),
                    fPath( aPath ),
                    fRotatedMeshCount( 64 )
            {
            }
            virtual ~KGRotatedClosedPathSpace()
            {
            }

        public:
            std::shared_ptr< XPathType > Path()
            {
                return fPath;
            }
            const std::shared_ptr< XPathType > Path() const
            {
                return fPath;
            }

            void RotatedMeshCount( const unsigned int& aCount )
            {
                fRotatedMeshCount = aCount;
                return;
            }
            const unsigned int& RotatedMeshCount() const
            {
                return fRotatedMeshCount;
            }

        protected:
            mutable std::shared_ptr< XPathType > fPath;
            mutable unsigned int fRotatedMeshCount;

        public:
            virtual void VolumeInitialize( BoundaryContainer& aBoundaryContainer ) const
            {
                auto tJacket = std::make_shared<KGRotatedPathSurface< XPathType >>( fPath );
                tJacket->RotatedMeshCount( fRotatedMeshCount );
                tJacket->SetName( "jacket" );
                aBoundaryContainer.push_back( tJacket );

                return;
            }
            virtual void VolumeAccept( KGVisitor* aVisitor )
            {
                shapemsg_debug( "rotated closed path volume named <" << GetName() << "> is receiving a visitor" << eom );
                typename KGRotatedClosedPathSpace::Visitor* tRotatedClosedPathSpaceVisitor = dynamic_cast< typename KGRotatedClosedPathSpace::Visitor* >( aVisitor );
                if( tRotatedClosedPathSpaceVisitor != NULL )
                {
                    shapemsg_debug( "rotated closed path volume named <" << GetName() << "> is accepting a visitor" << eom );
                    tRotatedClosedPathSpaceVisitor->VisitRotatedClosedPathSpace( this );
                    return;
                }
                KGVolume::VolumeAccept( aVisitor );
                return;
            }
            bool VolumeOutside( const KThreeVector& aPoint ) const
            {
                KTwoVector tZRPoint = aPoint.ProjectZR();
                return fPath->Above( tZRPoint );
            }
            KThreeVector VolumePoint( const KThreeVector& aPoint ) const
            {
                KTwoVector tZRPoint = aPoint.ProjectZR();
                KTwoVector tZRNearest = fPath->Point( tZRPoint );
                double tAngle = aPoint.AzimuthalAngle();
                return KThreeVector( cos( tAngle ) * tZRNearest.R(), sin( tAngle ) * tZRNearest.R(), tZRNearest.Z() );
            }
            KThreeVector VolumeNormal( const KThreeVector& aPoint ) const
            {
                KTwoVector tZRPoint = aPoint.ProjectZR();
                KTwoVector tZRNormal = fPath->Normal( tZRPoint );
                double tAngle = aPoint.AzimuthalAngle();
                return KThreeVector( cos( tAngle ) * tZRNormal.R(), sin( tAngle ) * tZRNormal.R(), tZRNormal.Z() );
            }

        private:
            static KGPlanarClosedPath* CompilerCheck()
            {
                XPathType* tPath = NULL;
                return tPath;
            }
    };

}

#endif
