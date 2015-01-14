#ifndef KGSHELLPATHSURFACE_HH_
#define KGSHELLPATHSURFACE_HH_

#include "KGArea.hh"
#include "KGPlanarPath.hh"
#include "KGShapeMessage.hh"

namespace KGeoBag
{

    template< class XPathType >
    class KGShellPathSurface :
    public KGArea
    {
    public:
        class Visitor
        {
        public:
            Visitor();
            virtual ~Visitor();

            virtual void VisitShellPathSurface( KGShellPathSurface* aShellPathSurface ) = 0;
        };

    public:
        KGShellPathSurface() :
        KGArea(),
        fPath( new XPathType() ),
        fSign( 1. ),
        fShellMeshCount( 64 ),
        fShellMeshPower( 1. ),
        fAngleStart(0),
        fAngleStop(360)
        {
            CompilerCheck();
        }
        KGShellPathSurface( const KGShellPathSurface< XPathType >& aCopy ) :
        KGArea( aCopy ),
        fPath( aCopy.fPath ),
        fSign( 1. ),
        fShellMeshCount( aCopy.fShellMeshCount ),
        fShellMeshPower( aCopy.fShellMeshPower ),
        fAngleStart(aCopy.fAngleStart),
        fAngleStop(aCopy.fAngleStop)
        {
        }
        KGShellPathSurface( KSmartPointer< XPathType > aPath ) :
        KGArea(),
        fPath( aPath ),
        fSign( 1. ),
        fShellMeshCount( 64 ),
        fShellMeshPower( 1. ),
        fAngleStart(0),
        fAngleStop(360)
        {
        }
        virtual ~KGShellPathSurface()
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

        void ShellMeshCount( const unsigned int& aCount )
        {
            fShellMeshCount = aCount;
        }
        const unsigned int& ShellMeshCount() const
        {
            return fShellMeshCount;
        }

        void ShellMeshPower( const double& aPower )
        {
            fShellMeshPower = aPower;
            return;
        }
        const double& ShellMeshPower() const
        {
            return fShellMeshPower;
        }

        void AngleStart( const double& aAngle )
        {
            fAngleStart = aAngle;
        }
        const double& AngleStart() const
        {
            return fAngleStart;
        }

        void AngleStop( const double& aAngle )
        {
            fAngleStop = aAngle;
        }
        const double& AngleStop() const
        {
            return fAngleStop;
        }

    protected:
        mutable KSmartPointer< XPathType > fPath;
        mutable double fSign;
        mutable unsigned int fShellMeshCount;
        mutable double fShellMeshPower;
        mutable double fAngleStart;
        mutable double fAngleStop;

    public:
        virtual void AreaInitialize() const
        {
            return;
        }
        virtual void AreaAccept( KGVisitor* aVisitor )
        {
            shapemsg_debug( "shell path area named <" << GetName() << "> is receiving a visitor" << eom );
            typename KGShellPathSurface< XPathType >::Visitor* tShellPathSurfaceVisitor = dynamic_cast< typename KGShellPathSurface< XPathType >::Visitor* >( aVisitor );
            if( tShellPathSurfaceVisitor != NULL )
            {
                shapemsg_debug( "shell path area named <" << GetName() << "> is accepting a visitor" << eom );
                tShellPathSurfaceVisitor->VisitShellPathSurface( this );
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
