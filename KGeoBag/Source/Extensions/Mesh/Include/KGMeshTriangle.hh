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

            virtual double NearestDistance(const KThreeVector& aPoint) const;
            virtual KThreeVector NearestPoint(const KThreeVector& aPoint) const;
            virtual KThreeVector NearestNormal(const KThreeVector& aPoint) const;
            virtual bool NearestIntersection(const KThreeVector& aStart, const KThreeVector& anEnd, KThreeVector& anIntersection) const;

            virtual KGPointCloud<KGMESH_DIM> GetPointCloud() const;
            virtual unsigned int GetNumberOfEdges() const {return 3;};
            virtual void GetEdge(KThreeVector& start, KThreeVector& end, unsigned int index) const;

            //assignment
            inline KGMeshTriangle& operator=(const KGMeshTriangle& t)
            {
                if(&t != this)
                {
                    fA = t.fA;
                    fB = t.fB;
                    fP0 = t.fP0;
                    fN1 = t.fN1;
                    fN2 = t.fN2;
                }
                return *this;
            }

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
            const KThreeVector GetN3() const
            {
                return fN1.Cross(fN2);
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
            void GetN3( KThreeVector& n3 ) const
            {
                n3 = fN1.Cross(fN2);
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

            bool SameSide(KThreeVector point, KThreeVector A, KThreeVector B, KThreeVector C) const;

            KThreeVector NearestPointOnLineSegment(const KThreeVector& a, const KThreeVector& b, const KThreeVector& point) const;

            double fA;
            double fB;
            KThreeVector fP0;
            KThreeVector fN1;
            KThreeVector fN2;

    };
}

#endif /* KGMESHTRIANGLE_DEF */
