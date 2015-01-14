#ifndef KGEXTRUDEDCLOSEDPATHSPACE_HH_
#define KGEXTRUDEDCLOSEDPATHSPACE_HH_

#include "KGVolume.hh"
#include "KGPlanarClosedPath.hh"
#include "KGExtrudedPathSurface.hh"
#include "KGFlattenedClosedPathSurface.hh"
#include "KGShapeMessage.hh"

namespace KGeoBag
{

    template< class XPathType >
    class KGExtrudedClosedPathSpace :
        public KGVolume
    {
        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitExtrudedClosedPathSpace( KGExtrudedClosedPathSpace* aExtrudedClosedPathSpace ) = 0;
            };

        public:
            KGExtrudedClosedPathSpace() :
                    KGVolume(),
                    fPath( new XPathType() ),
                    fZMin( 0. ),
                    fZMax( 0. ),
                    fExtrudedMeshCount( 1 ),
                    fExtrudedMeshPower( 1. ),
                    fFlattenedMeshCount( 1 ),
                    fFlattenedMeshPower( 1. )
            {
                CompilerCheck();
            }
            KGExtrudedClosedPathSpace( const KGExtrudedClosedPathSpace< XPathType >& aCopy ) :
                    KGVolume( aCopy ),
                    fPath( aCopy.fPath->Clone() ),
                    fZMin( aCopy.fZMin ),
                    fZMax( aCopy.fZMax ),
                    fExtrudedMeshCount( aCopy.fExtrudedMeshCount ),
                    fExtrudedMeshPower( aCopy.fExtrudedMeshPower ),
                    fFlattenedMeshCount( aCopy.fFlattenedMeshCount ),
                    fFlattenedMeshPower( aCopy.fFlattenedMeshPower )
            {
            }
            KGExtrudedClosedPathSpace( const KSmartPointer< XPathType >& aPath ) :
                    KGVolume(),
                    fPath( aPath ),
                    fZMin( 0. ),
                    fZMax( 0. ),
                    fExtrudedMeshCount( 1 ),
                    fExtrudedMeshPower( 1. ),
                    fFlattenedMeshCount( 1 ),
                    fFlattenedMeshPower( 1. )
            {
            }
            virtual ~KGExtrudedClosedPathSpace()
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

            void ZMin( const double& aZMin )
            {
                fZMin = aZMin;
                return;
            }
            const double& ZMin() const
            {
                return fZMin;
            }

            void ZMax( const double& aZMax )
            {
                fZMax = aZMax;
                return;
            }
            const double& ZMax() const
            {
                return fZMax;
            }

            void ExtrudedMeshCount( const unsigned int& aCount )
            {
                fExtrudedMeshCount = aCount;
                return;
            }
            const unsigned int& ExtrudedMeshCount() const
            {
                return fExtrudedMeshCount;
            }

            void ExtrudedMeshPower( const double& aPower )
            {
                fExtrudedMeshPower = aPower;
                return;
            }
            const double& ExtrudedMeshPower() const
            {
                return fExtrudedMeshPower;
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
            mutable double fZMin;
            mutable double fZMax;
            mutable unsigned int fExtrudedMeshCount;
            mutable double fExtrudedMeshPower;
            mutable unsigned int fFlattenedMeshCount;
            mutable double fFlattenedMeshPower;

        public:
            virtual void VolumeInitialize( BoundaryContainer& aBoundaryContainer ) const
            {
                KGFlattenedClosedPathSurface< XPathType >* tTop = new KGFlattenedClosedPathSurface< XPathType >( fPath );
                tTop->Z( fZMax );
                tTop->FlattenedMeshCount( fFlattenedMeshCount );
                tTop->FlattenedMeshPower( fFlattenedMeshPower );
                tTop->SetName( "top" );
                aBoundaryContainer.push_back( tTop );

                KGExtrudedPathSurface< XPathType >* tJacket = new KGExtrudedPathSurface< XPathType >( fPath );
                tJacket->ExtrudedMeshCount( fExtrudedMeshCount );
                tJacket->SetName( "jacket" );
                aBoundaryContainer.push_back( tJacket );

                KGFlattenedClosedPathSurface< XPathType >* tBottom = new KGFlattenedClosedPathSurface< XPathType >( fPath );
                tBottom->Z( fZMin );
                tBottom->FlattenedMeshCount( fFlattenedMeshCount );
                tBottom->FlattenedMeshPower( fFlattenedMeshPower );
                tBottom->SetName( "bottom" );
                aBoundaryContainer.push_back( tBottom );

                return;
            }
            virtual void VolumeAccept( KGVisitor* aVisitor )
            {
                shapemsg_debug( "extruded closed path volume named <" << GetName() << "> is receiving a visitor" << eom );
                typename KGExtrudedClosedPathSpace::Visitor* tExtrudedClosedPathSpaceVisitor = dynamic_cast< typename KGExtrudedClosedPathSpace::Visitor* >( aVisitor );
                if( tExtrudedClosedPathSpaceVisitor != NULL )
                {
                    shapemsg_debug( "extruded closed path volume named <" << GetName() << "> is accepting a visitor" << eom );
                    tExtrudedClosedPathSpaceVisitor->VisitExtrudedClosedPathSpace( this );
                    return;
                }
                KGVolume::VolumeAccept( aVisitor );
                return;
            }
            virtual bool VolumeOutside( const KThreeVector& aPoint ) const
            {
                KTwoVector tXYPoint = aPoint.ProjectXY();
                double tZPoint = aPoint.Z();

                KTwoVector tJacketPoint = fPath->Point( tXYPoint ) - tXYPoint;

                KTwoVector tCapPoint( 0., 0. );
                if( fPath->Above( tXYPoint ) == true )
                {
                    tCapPoint = tJacketPoint;
                }

                double tJacketZ = 0.;
                if( tZPoint > fZMax )
                {
                    tJacketZ = fZMax - tZPoint;
                }
                if( tZPoint < fZMin )
                {
                    tJacketZ = fZMin - tZPoint;
                }

                double tTopZ = tZPoint - fZMax;
                double tBottomZ = fZMin - tZPoint;

                double tJacketDistanceSquared = tJacketPoint.MagnitudeSquared() + tJacketZ * tJacketZ;
                double tTopDistanceSquared = tCapPoint.MagnitudeSquared() + tTopZ * tTopZ;
                double tBottomDistanceSquared = tCapPoint.MagnitudeSquared() + tBottomZ * tBottomZ;

                if( tTopDistanceSquared < tJacketDistanceSquared )
                {
                    if( tTopDistanceSquared < tBottomDistanceSquared )
                    {
                        if( tTopZ > 0. )
                        {
                            return true;
                        }
                        return false;
                    }
                }
                if( tBottomDistanceSquared < tJacketDistanceSquared )
                {
                    if( tBottomDistanceSquared < tTopDistanceSquared )
                    {
                        if( tBottomZ < 0. )
                        {
                            return true;
                        }
                        return false;
                    }
                }
                return fPath->Above( tXYPoint );
            }
            virtual KThreeVector VolumePoint( const KThreeVector& aPoint ) const
            {
                KTwoVector tXYPoint = aPoint.ProjectXY();
                double tZPoint = aPoint.Z();

                KTwoVector tJacketPoint = fPath->Point( tXYPoint ) - tXYPoint;

                KTwoVector tCapPoint( 0., 0. );
                if( fPath->Above( tXYPoint ) == true )
                {
                    tCapPoint = tJacketPoint;
                }

                double tJacketZ = 0.;
                if( tZPoint > fZMax )
                {
                    tJacketZ = fZMax - tZPoint;
                }
                if( tZPoint < fZMin )
                {
                    tJacketZ = fZMin - tZPoint;
                }

                double tTopZ = tZPoint - fZMax;
                double tBottomZ = fZMin - tZPoint;

                double tJacketDistanceSquared = tJacketPoint.MagnitudeSquared() + tJacketZ * tJacketZ;
                double tTopDistanceSquared = tCapPoint.MagnitudeSquared() + tTopZ * tTopZ;
                double tBottomDistanceSquared = tCapPoint.MagnitudeSquared() + tBottomZ * tBottomZ;

                KTwoVector tXYNearest;
                double tZNearest;
                if( tTopDistanceSquared < tJacketDistanceSquared )
                {
                    if( tTopDistanceSquared < tBottomDistanceSquared )
                    {
                        tXYNearest = tXYPoint + tCapPoint;
                        tZNearest = tZPoint + tTopZ;
                        return KThreeVector( tXYNearest.X(), tXYNearest.Y(), tZNearest );
                    }
                }
                if( tBottomDistanceSquared < tJacketDistanceSquared )
                {
                    if( tBottomDistanceSquared < tTopDistanceSquared )
                    {
                        tXYNearest = tXYPoint + tCapPoint;
                        tZNearest = tZPoint + tBottomZ;
                        return KThreeVector( tXYNearest.X(), tXYNearest.Y(), tZNearest );
                    }
                }
                tXYNearest = tXYPoint + tJacketPoint;
                tZNearest = tZPoint + tJacketZ;
                return KThreeVector( tXYNearest.X(), tXYNearest.Y(), tZNearest );
            }
            virtual KThreeVector VolumeNormal( const KThreeVector& aPoint ) const
            {
                KTwoVector tXYPoint = aPoint.ProjectXY();
                double tZPoint = aPoint.Z();

                KTwoVector tJacketPoint = fPath->Point( tXYPoint ) - tXYPoint;

                KTwoVector tCapPoint( 0., 0. );
                if( fPath->Above( tXYPoint ) == true )
                {
                    tCapPoint = tJacketPoint;
                }

                double tJacketZ = 0.;
                if( tZPoint > fZMax )
                {
                    tJacketZ = fZMax - tZPoint;
                }
                if( tZPoint < fZMin )
                {
                    tJacketZ = fZMin - tZPoint;
                }

                double tTopZ = tZPoint - fZMax;
                double tBottomZ = fZMin - tZPoint;

                double tJacketDistanceSquared = tJacketPoint.MagnitudeSquared() + tJacketZ * tJacketZ;
                double tTopDistanceSquared = tCapPoint.MagnitudeSquared() + tTopZ * tTopZ;
                double tBottomDistanceSquared = tCapPoint.MagnitudeSquared() + tBottomZ * tBottomZ;

                KTwoVector tXYNormal( 0., 0. );
                double tZNormal = 0.;
                if( tTopDistanceSquared < tJacketDistanceSquared )
                {
                    if( tTopDistanceSquared < tBottomDistanceSquared )
                    {
                        tZNormal = 1.;
                        return KThreeVector( tXYNormal.X(), tXYNormal.Y(), tZNormal );
                    }
                }
                if( tBottomDistanceSquared < tJacketDistanceSquared )
                {
                    if( tBottomDistanceSquared < tTopDistanceSquared )
                    {
                        tZNormal = 1.;
                        return KThreeVector( tXYNormal.X(), tXYNormal.Y(), tZNormal );
                    }
                }
                tXYNormal = fPath->Normal( tXYPoint );
                return KThreeVector( tXYNormal.X(), tXYNormal.Y(), tZNormal );
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
