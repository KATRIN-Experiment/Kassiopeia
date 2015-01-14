#ifndef KGROTATEDPATHSURFACE_HH_
#define KGROTATEDPATHSURFACE_HH_

#include "KGArea.hh"
#include "KGPlanarPath.hh"
#include "KGShapeMessage.hh"

namespace KGeoBag
{

    template< class XPathType >
    class KGRotatedPathSurface :
        public KGArea
    {
        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitRotatedPathSurface( KGRotatedPathSurface* aRotatedPathSurface ) = 0;
            };

        public:
            KGRotatedPathSurface() :
                    KGArea(),
                    fPath( new XPathType() ),
                    fSign( 1. ),
                    fRotatedMeshCount( 64 )
            {
                CompilerCheck();
            }
            KGRotatedPathSurface( const KGRotatedPathSurface< XPathType >& aCopy ) :
                    KGArea( aCopy ),
                    fPath( aCopy.fPath ),
                    fSign( 1. ),
                    fRotatedMeshCount( aCopy.fRotatedMeshCount )
            {
            }
            KGRotatedPathSurface( KSmartPointer< XPathType > aPath ) :
                    KGArea(),
                    fPath( aPath ),
                    fSign( 1. ),
                    fRotatedMeshCount( 64 )
            {
            }
            virtual ~KGRotatedPathSurface()
            {
            }

        public:
            KSmartPointer< XPathType >& Path()
            {
                return fPath;
            }
            const KSmartPointer< XPathType >& Path() const
            {
                return fPath;
            }

            void Sign( const double& aSign )
            {
                fSign = aSign / fabs( aSign );
                return;
            }
            const double& Sign() const
            {
                return fSign;
            }

            void RotatedMeshCount( const unsigned int& aCount )
            {
                fRotatedMeshCount = aCount;
            }
            const unsigned int& RotatedMeshCount() const
            {
                return fRotatedMeshCount;
            }

        protected:
            mutable KSmartPointer< XPathType > fPath;
            mutable double fSign;
            mutable unsigned int fRotatedMeshCount;

        public:
            virtual void AreaInitialize() const
            {
                return;
            }
            virtual void AreaAccept( KGVisitor* aVisitor )
            {
                shapemsg_debug( "rotated path area named <" << GetName() << "> is receiving a visitor" << eom );
                typename KGRotatedPathSurface< XPathType >::Visitor* tRotatedPathSurfaceVisitor = dynamic_cast< typename KGRotatedPathSurface< XPathType >::Visitor* >( aVisitor );
                if( tRotatedPathSurfaceVisitor != NULL )
                {
                    shapemsg_debug( "rotated path area named <" << GetName() << "> is accepting a visitor" << eom );
                    tRotatedPathSurfaceVisitor->VisitRotatedPathSurface( this );
                }
                return;
            }
            virtual bool AreaAbove( const KThreeVector& aPoint ) const
            {
                KTwoVector tZRPoint = aPoint.ProjectZR();
                bool tZRAbove = fPath->Above( tZRPoint );
                if( (tZRAbove == true ) && ( fSign > 0. ) )
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            virtual KThreeVector AreaPoint( const KThreeVector& aPoint ) const
            {
                KTwoVector tZRPoint = aPoint.ProjectZR();
                KTwoVector tZRNearest = fPath->Point( tZRPoint );
                double tAngle = aPoint.AzimuthalAngle();
                return KThreeVector( cos( tAngle ) * tZRNearest.R(), sin( tAngle ) * tZRNearest.R(), tZRNearest.Z() );
            }
            virtual KThreeVector AreaNormal( const KThreeVector& aPoint ) const
            {
                KTwoVector tZRPoint = aPoint.ProjectZR();
                KTwoVector tZRNormal = fPath->Normal( tZRPoint );
                double tAngle = aPoint.AzimuthalAngle();
                return fSign * KThreeVector( cos( tAngle ) * tZRNormal.R(), sin( tAngle ) * tZRNormal.R(), tZRNormal.Z() );
            }

        private:
            static KGPlanarPath* CompilerCheck()
            {
                XPathType* tPath = NULL;
                return tPath;
            }
    };

}

#endif
