#ifndef KGMESHTRIANGLE_DEF
#define KGMESHTRIANGLE_DEF

#include "KThreeVector.hh"

#include "KGMeshElement.hh"

namespace KGeoBag
{
    class KGMeshTriangle :
        public KGMeshElement
    {
        public:
            KGMeshTriangle( const double& a, const double& b, const KThreeVector& p0, const KThreeVector& n1, const KThreeVector& n2 );
            KGMeshTriangle( const KThreeVector& p0, const KThreeVector& p1, const KThreeVector& p2 );
            virtual ~KGMeshTriangle();

            double Area() const;
            double Aspect() const;
            void Transform( const KTransformation& transform );

            double GetA() const
            {
                return fA;
            }
            double GetB() const
            {
                return fB;
            }
            const KThreeVector& GetP0() const
            {
                return fP0;
            }
            const KThreeVector& GetN1() const
            {
                return fN1;
            }
            const KThreeVector& GetN2() const
            {
                return fN2;
            }
            const KThreeVector GetP1() const
            {
                return fP0 + fN1 * fA;
            }
            const KThreeVector GetP2() const
            {
                return fP0 + fN2 * fB;
            }
            void GetP0( KThreeVector& p0 ) const
            {
                p0 = fP0;
            }
            void GetN1( KThreeVector& n1 ) const
            {
                n1 = fN1;
            }
            void GetN2( KThreeVector& n2 ) const
            {
                n2 = fN2;
            }
            void GetP1( KThreeVector& p1 ) const
            {
                p1 = fP0 + fN1 * fA;
            }
            void GetP2( KThreeVector& p2 ) const
            {
                p2 = fP0 + fN2 * fB;
            }

        protected:
            double fA;
            double fB;
            KThreeVector fP0;
            KThreeVector fN1;
            KThreeVector fN2;

    };
}

#endif /* KGMESHTRIANGLE_DEF */
