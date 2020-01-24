#ifndef KGMESHRECTANGLE_DEF
#define KGMESHRECTANGLE_DEF

#include "KGMeshElement.hh"

#include "KThreeVector.hh"

namespace KGeoBag
{
    class KGMeshRectangle :
        public KGMeshElement
    {
        public:
            KGMeshRectangle( const double& a, const double& b, const KThreeVector& p0, const KThreeVector& n1, const KThreeVector& n2 );
            KGMeshRectangle( const KThreeVector& p0, const KThreeVector& p1, const KThreeVector& /*p2*/, const KThreeVector& p3 );
            KGMeshRectangle( const KGMeshRectangle& r );
            virtual ~KGMeshRectangle();

            double Area() const;
            double Aspect() const;
            void Transform( const KTransformation& transform );

            virtual double NearestDistance(const KThreeVector& aPoint) const;
            virtual KThreeVector NearestPoint(const KThreeVector& aPoint) const;
            virtual KThreeVector NearestNormal(const KThreeVector& aPoint) const;
            virtual bool NearestIntersection(const KThreeVector& aStart, const KThreeVector& anEnd, KThreeVector& anIntersection) const;

            virtual KGPointCloud<KGMESH_DIM> GetPointCloud() const;
            virtual unsigned int GetNumberOfEdges() const {return 4;};
            virtual void GetEdge(KThreeVector& start, KThreeVector& end, unsigned int index) const;

            //assignment
            inline KGMeshRectangle& operator=(const KGMeshRectangle& r)
            {
                if(&r != this)
                {
                    fA = r.fA;
                    fB = r.fB;
                    fP0 = r.fP0;
                    fN1 = r.fN1;
                    fN2 = r.fN2;
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
                return fP0 + fN1 * fA + fN2 * fB;
            }
            const KThreeVector GetP3() const
            {
                return fP0 + fN2 * fB;
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
            void GetP0( KThreeVector& p0 ) const
            {
                p0 = fP0;
            }
            void GetP1( KThreeVector& p1 ) const
            {
                p1 = fP0 + fN1 * fA;
            }
            void GetP2( KThreeVector& p2 ) const
            {
                p2 = fP0 + fN1 * fA + fN2 * fA;
            }
            void GetP3( KThreeVector& p3 ) const
            {
                p3 = fP0 + fN2 * fB;
            }

        protected:

            bool SameSide(KThreeVector point, KThreeVector A, KThreeVector B, KThreeVector C) const;


            double fA;
            double fB;
            KThreeVector fP0;
            KThreeVector fN1;
            KThreeVector fN2;
    };
}

#endif /* KGMESHRECTANGLE_DEF */
