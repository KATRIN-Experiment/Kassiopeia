#ifndef KGCYLINDERSURFACE_HH_
#define KGCYLINDERSURFACE_HH_

#include "KGRotatedLineSegmentSurface.hh"

namespace KGeoBag
{

    class KGCylinderSurface :
        public KGRotatedLineSegmentSurface
    {
        public:
            KGCylinderSurface();
            virtual ~KGCylinderSurface();

        public:
            void Z1( const double& aZ1 );
            void Z2( const double& aZ2 );
            void R( const double& anR );
            void LongitudinalMeshCount( const unsigned int& aLongitudinalMeshCount );
            void LongitudinalMeshPower( const double& aLongitudinalMeshPower );
            void AxialMeshCount( const unsigned int& anAxialMeshCount );

            const double& Z1() const;
            const double& Z2() const;
            const double& R() const;
            const unsigned int& LongitudinalMeshCount() const;
            const double& LongitudinalMeshPower() const;
            const unsigned int& AxialMeshCount() const;

        private:
            double fZ1;
            double fZ2;
            double fR;
            unsigned int fLongitudinalMeshCount;
            double fLongitudinalMeshPower;
            unsigned int fAxialMeshCount;

        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitCylinderSurface( KGCylinderSurface* aCylinderSurface ) = 0;
            };

        public:
            virtual void AreaInitialize() const;
            virtual void AreaAccept( KGVisitor* aVisitor );
    };

}

#endif
