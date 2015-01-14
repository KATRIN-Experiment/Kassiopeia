#ifndef KGCYLINDERSPACE_HH_
#define KGCYLINDERSPACE_HH_

#include "KGRotatedLineSegmentSpace.hh"

namespace KGeoBag
{

    class KGCylinderSpace :
        public KGRotatedLineSegmentSpace
    {
        public:
            KGCylinderSpace();
            virtual ~KGCylinderSpace();

        public:
            void Z1( const double& aZ1 );
            void Z2( const double& aZ2 );
            void R( const double& anR );
            void LongitudinalMeshCount( const unsigned int& aLongitudinalMeshCount );
            void LongitudinalMeshPower( const double& aLongitudinalMeshPower );
            void RadialMeshCount( const unsigned int& aRadialMeshCount );
            void RadialMeshPower( const double& aRadialMeshPower );
            void AxialMeshCount( const unsigned int& anAxialMeshCount );

            const double& Z1() const;
            const double& Z2() const;
            const double& R() const;
            const unsigned int& LongitudinalMeshCount() const;
            const double& LongitudinalMeshPower() const;
            const unsigned int& RadialMeshCount() const;
            const double& RadialMeshPower() const;
            const unsigned int& AxialMeshCount() const;

        private:
            double fZ1;
            double fZ2;
            double fR;
            unsigned int fLongitudinalMeshCount;
            double fLongitudinalMeshPower;
            unsigned int fRadialMeshCount;
            double fRadialMeshPower;
            unsigned int fAxialMeshCount;

        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitCylinderSpace( KGCylinderSpace* aCylinderSpace ) = 0;
            };

        public:
            virtual void VolumeInitialize( BoundaryContainer& aBoundaryContainer ) const;
            virtual void VolumeAccept( KGVisitor* aVisitor );

    };

}

#endif
