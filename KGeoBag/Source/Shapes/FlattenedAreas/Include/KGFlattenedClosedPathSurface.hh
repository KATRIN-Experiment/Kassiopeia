#ifndef KGFLATTENEDCLOSEDPATHSURFACE_HH_
#define KGFLATTENEDCLOSEDPATHSURFACE_HH_

#include "KGArea.hh"
#include "KGPlanarClosedPath.hh"
#include "KGShapeMessage.hh"

namespace KGeoBag
{

    template< class XPathType >
    class KGFlattenedClosedPathSurface :
        public KGArea
    {
        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitFlattenedClosedPathSurface( KGFlattenedClosedPathSurface< XPathType >* aFlattenedClosedPathSurface ) = 0;
            };

        public:
            KGFlattenedClosedPathSurface() :
                    KGArea(),
                    fPath( new XPathType() ),
                    fZ( 0. ),
                    fSign( 1. ),
                    fFlattenedMeshCount( 8 ),
                    fFlattenedMeshPower( 1. )
            {
                CompilerCheck();
            }
            KGFlattenedClosedPathSurface( const KGFlattenedClosedPathSurface< XPathType >& aCopy ) :
                    KGArea( aCopy ),
                    fPath( aCopy.fPath ),
                    fZ( aCopy.fZ ),
                    fSign( aCopy.fSign ),
                    fFlattenedMeshCount( aCopy.fFlattenedMeshCount ),
                    fFlattenedMeshPower( aCopy.fFlattenedMeshPower )
            {
            }
            KGFlattenedClosedPathSurface( const KSmartPointer< XPathType >& aPath ) :
                    KGArea(),
                    fPath( aPath ),
                    fZ( 0. ),
                    fSign( 1. ),
                    fFlattenedMeshCount( 8 ),
                    fFlattenedMeshPower( 1. )
            {
            }
            virtual ~KGFlattenedClosedPathSurface()
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

            void Z( const double& aZ )
            {
                fZ = aZ;
                return;
            }
            const double& Z() const
            {
                return fZ;
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

            void FlattenedMeshCount( const unsigned int& aCount )
            {
                fFlattenedMeshCount = aCount;
                return;
            }
            const unsigned int& FlattenedMeshCount() const
            {
                return fFlattenedMeshCount;
            }

            void FlattenedMeshPower( const double& aPower )
            {
                fFlattenedMeshPower = aPower;
                return;
            }
            const double& FlattenedMeshPower() const
            {
                return fFlattenedMeshPower;
            }

        protected:
            mutable KSmartPointer< XPathType > fPath;
            mutable double fZ;
            mutable double fSign;
            mutable unsigned int fFlattenedMeshCount;
            mutable double fFlattenedMeshPower;

        public:
            virtual void AreaInitialize() const
            {
                return;
            }
            virtual void AreaAccept( KGVisitor* aVisitor )
            {
                shapemsg_debug( "flattened closed path area named <" << GetName() << "> is receiving a visitor" << eom )
                typename KGFlattenedClosedPathSurface< XPathType >::Visitor* tFlattenedClosedPathSurfaceVisitor = dynamic_cast< typename KGFlattenedClosedPathSurface< XPathType >::Visitor* >( aVisitor );
                if( tFlattenedClosedPathSurfaceVisitor != NULL )
                {
                    shapemsg_debug( "flattened closed path area named <" << GetName() << "> is accepting a visitor" << eom )
                    tFlattenedClosedPathSurfaceVisitor->VisitFlattenedClosedPathSurface( this );
                }
                return;
            }
            virtual bool AreaAbove( const KThreeVector& aPoint ) const
            {
                double tZ = aPoint.Z();

                if( ( (tZ > fZ) && (fSign > 0.) ) || ( !(tZ > fZ) && !(fSign > 0.) )  )
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
                KTwoVector tXYPoint = aPoint.ProjectXY();

                KTwoVector tXYNearest;
                if( fPath->Above( tXYPoint ) == true )
                {
                    tXYNearest = fPath->Point( tXYPoint );
                }
                else
                {
                    tXYNearest = tXYPoint;
                }

                return KThreeVector( tXYNearest.X(), tXYNearest.Y(), fZ );
            }
            virtual KThreeVector AreaNormal( const KThreeVector& /*aPoint*/) const
            {
                return KThreeVector( 0., 0., fSign );
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
