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

            virtual double NearestDistance(const KThreeVector& aPoint) const;
            virtual KThreeVector NearestPoint(const KThreeVector& aPoint) const;
            virtual KThreeVector NearestNormal(const KThreeVector& aPoint) const;
            virtual bool NearestIntersection(const KThreeVector& aStart, const KThreeVector& anEnd, KThreeVector& anIntersection) const;

            virtual KGPointCloud<KGMESH_DIM> GetPointCloud() const;
            virtual unsigned int GetNumberOfEdges() const {return 1;};
            virtual void GetEdge(KThreeVector& start, KThreeVector& end, unsigned int /*index*/) const;

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

            double ClosestApproach(const KThreeVector& aStart, const KThreeVector& anEnd) const;

            KThreeVector fP0;
            KThreeVector fP1;
            double fDiameter;
    };
}

#endif /* KGMESHTRIANGLE_DEF */
