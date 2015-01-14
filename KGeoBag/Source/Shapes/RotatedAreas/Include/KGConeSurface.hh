#ifndef KGCONESURFACE_HH_
#define KGCONESURFACE_HH_

#include "KGRotatedLineSegmentSurface.hh"

namespace KGeoBag
{

    class KGConeSurface :
        public KGRotatedLineSegmentSurface
    {
        public:
            KGConeSurface();
            virtual ~KGConeSurface();

        public:
            void ZA( const double& aZA );
            void ZB( const double& aZB );
            void RB( const double& anRB );
            void LongitudinalMeshCount( const unsigned int& aLongitudinalMeshCount );
            void LongitudinalMeshPower( const double& aLongitudinalMeshPower );
            void AxialMeshCount( const unsigned int& anAxialMeshCount );

            const double& ZA() const;
            const double& ZB() const;
            const double& RB() const;
            const unsigned int& LongitudinalMeshCount() const;
            const double& LongitudinalMeshPower() const;
            const unsigned int& AxialMeshCount() const;

        private:
            double fZA;
            double fZB;
            double fRB;
            unsigned int fLongitudinalMeshCount;
            double fLongitudinalMeshPower;
            unsigned int fAxialMeshCount;

        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitConeSurface( KGConeSurface* aConeSurface ) = 0;
            };

        public:
            virtual void AreaInitialize() const;
            virtual void AreaAccept( KGVisitor* aVisitor );
    };

}

#endif
