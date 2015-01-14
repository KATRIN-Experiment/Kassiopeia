#ifndef KGMESHWIRE_DEF
#define KGMESHWIRE_DEF

#include "KThreeVector.hh"

#include "KGMeshElement.hh"

namespace KGeoBag
{
    class KGMeshWire :
        public KGMeshElement
    {
        public:
            KGMeshWire( const KThreeVector& p0, const KThreeVector& p1, const double& diameter );
            virtual ~KGMeshWire();

            double Area() const;
            double Aspect() const;
            void Transform( const KTransformation& transform );

            const KThreeVector& GetP0() const
            {
                return fP0;
            }
            const KThreeVector& GetP1() const
            {
                return fP1;
            }
            double GetDiameter() const
            {
                return fDiameter;
            }
            void GetP0( KThreeVector& p0 ) const
            {
                p0 = fP0;
            }
            void GetP1( KThreeVector& p1 ) const
            {
                p1 = fP1;
            }

        protected:
            KThreeVector fP0;
            KThreeVector fP1;
            double fDiameter;
    };
}

#endif /* KGMESHTRIANGLE_DEF */
