#ifndef KGCUTCONESURFACE_HH_
#define KGCUTCONESURFACE_HH_

#include "KGRotatedLineSegmentSurface.hh"

namespace KGeoBag
{

    class KGCutConeSurface :
        public KGRotatedLineSegmentSurface
    {
        public:
            KGCutConeSurface();
            virtual ~KGCutConeSurface();

        public:
            void Z1( const double& aZ1 );
            void Z2( const double& aZ2 );
            void R1( const double& anR1 );
            void R2( const double& anR2 );
            void LongitudinalMeshCount( const unsigned int& aLongitudinalMeshCount );
            void LongitudinalMeshPower( const double& aLongitudinalMeshPower );
            void AxialMeshCount( const unsigned int& anAxialMeshCount );

            const double& Z1() const;
            const double& Z2() const;
            const double& R1() const;
            const double& R2() const;
            const unsigned int& LongitudinalMeshCount() const;
            const double& LongitudinalMeshPower() const;
            const unsigned int& AxialMeshCount() const;

        private:
            double fZ1;
            double fR1;
            double fZ2;
            double fR2;
            unsigned int fLongitudinalMeshCount;
            double fLongitudinalMeshPower;
            unsigned int fAxialMeshCount;

        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitCutConeSurface( KGCutConeSurface* aCutConeSurface ) = 0;
            };

        public:
            virtual void AreaInitialize() const;
            virtual void AreaAccept( KGVisitor* aVisitor );
    };

}

#endif
