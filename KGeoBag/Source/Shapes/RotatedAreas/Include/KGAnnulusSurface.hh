#ifndef KGANNULUSSURFACE_HH_
#define KGANNULUSSURFACE_HH_

#include "KGRotatedLineSegmentSurface.hh"

namespace KGeoBag
{

    class KGAnnulusSurface :
        public KGRotatedLineSegmentSurface
    {
        public:
            KGAnnulusSurface();
            virtual ~KGAnnulusSurface();

        public:
            void Z( const double& aZ );
            void R1( const double& anR );
            void R2( const double& anR );
            void RadialMeshCount( const unsigned int& aRadialMeshCount );
            void RadialMeshPower( const double& aRadialMeshPower );
            void AxialMeshCount( const unsigned int& anAxialMeshCount );


            const double& Z() const;
            const double& R1() const;
            const double& R2() const;
            const unsigned int& RadialMeshCount() const;
            const double& RadialMeshPower() const;
            const unsigned int& AxialMeshCount() const;

        private:
            double fZ;
            double fR1;
            double fR2;
            unsigned int fRadialMeshCount;
            double fRadialMeshPower;
            unsigned int fAxialMeshCount;

        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitAnnulusSurface( KGAnnulusSurface* anAnnulusSurface ) = 0;
            };

        public:
            virtual void AreaInitialize() const;
            virtual void AreaAccept( KGVisitor* aVisitor );
    };

}

#endif
