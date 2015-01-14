#ifndef KGCUTCONETUBESPACE_HH_
#define KGCUTCONETUBESPACE_HH_

#include "KGRotatedPolyLoopSpace.hh"

namespace KGeoBag
{

    class KGCutConeTubeSpace :
        public KGRotatedPolyLoopSpace
    {
        public:
            KGCutConeTubeSpace();
            virtual ~KGCutConeTubeSpace();

        public:
            void Z1( const double& aZ1 );
            void Z2( const double& aZ2 );
            void R11( const double& anR1 );
            void R12( const double& anR2 );
            void R21( const double& anR1 );
            void R22( const double& anR2 );
            void RadialMeshCount( const unsigned int& aCount );
            void RadialMeshPower( const double& aPower );
            void LongitudinalMeshCount( const unsigned int& aCount );
            void LongitudinalMeshPower( const double& aPower );
            void AxialMeshCount( const unsigned int& anAxialMeshCount );

            const double& Z1() const;
            const double& Z2() const;
            const double& R11() const;
            const double& R12() const;
            const double& R21() const;
            const double& R22() const;
            const unsigned int& RadialMeshCount() const;
            const double& RadialMeshPower() const;
            const unsigned int& LongitudinalMeshCount() const;
            const double& LongitudinalMeshPower() const;
            const unsigned int& AxialMeshCount() const;

        private:
            double fZ1;
            double fZ2;
            double fR11;
            double fR12;
            double fR21;
            double fR22;
            unsigned int fRadialMeshCount;
            double fRadialMeshPower;
            unsigned int fLongitudinalMeshCount;
            double fLongitudinalMeshPower;
            unsigned int fAxialMeshCount;

        public:
            class Visitor
            {
                public:
                    Visitor();
                    virtual ~Visitor();

                    virtual void VisitCutConeTubeSpace( KGCutConeTubeSpace* aCutConeTubeSpace ) = 0;
            };

        public:
            virtual void VolumeInitialize( BoundaryContainer& aBoundaryContainer ) const;
            virtual void VolumeAccept( KGVisitor* aVisitor );
    };

}

#endif
