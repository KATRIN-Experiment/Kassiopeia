#ifndef KGCONESPACE_HH_
#define KGCONESPACE_HH_

#include "KGRotatedLineSegmentSpace.hh"

namespace KGeoBag
{

    class KGConeSpace :
        public KGRotatedLineSegmentSpace
    {
        public:
            KGConeSpace();
            virtual ~KGConeSpace();

        public:
            void ZA( const double& aZA );
            void ZB( const double& aZB );
            void RB( const double& anRB );
            void LongitudinalMeshCount( const unsigned int& aLongitudinalMeshCount );
            void LongitudinalMeshPower( const double& aLongitudinalMeshPower );
            void RadialMeshCount( const unsigned int& aRadialMeshCount );
            void RadialMeshPower( const double& aRadialMeshPower );
            void AxialMeshCount( const unsigned int& anAxialMeshCount );

            const double& ZA() const;
            const double& ZB() const;
            const double& RB() const;
            const unsigned int& LongitudinalMeshCount() const;
            const double& LongitudinalMeshPower() const;
            const unsigned int& RadialMeshCount() const;
            const double& RadialMeshPower() const;
            const unsigned int& AxialMeshCount() const;

        private:
            double fZA;
            double fZB;
            double fRB;
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

                    virtual void VisitConeSpace( KGConeSpace* aConeSpace ) = 0;
            };

        public:
            virtual void VolumeInitialize( BoundaryContainer& aBoundaryContainer ) const;
            virtual void VolumeAccept( KGVisitor* aVisitor );

    };

}

#endif
