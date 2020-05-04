#ifndef KGMESHWIRE_DEF
#define KGMESHWIRE_DEF

#include "KGMeshElement.hh"
#include "KThreeVector.hh"

namespace KGeoBag
{
class KGMeshWire : public KGMeshElement
{
  public:
    KGMeshWire(const KThreeVector& p0, const KThreeVector& p1, const double& diameter);
    ~KGMeshWire() override;

    double Area() const override;
    double Aspect() const override;
    void Transform(const KTransformation& transform) override;

    double NearestDistance(const KThreeVector& aPoint) const override;
    KThreeVector NearestPoint(const KThreeVector& aPoint) const override;
    KThreeVector NearestNormal(const KThreeVector& aPoint) const override;
    bool NearestIntersection(const KThreeVector& aStart, const KThreeVector& anEnd,
                             KThreeVector& anIntersection) const override;

    KGPointCloud<KGMESH_DIM> GetPointCloud() const override;
    unsigned int GetNumberOfEdges() const override
    {
        return 1;
    };
    void GetEdge(KThreeVector& start, KThreeVector& end, unsigned int /*index*/) const override;

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
    void GetP0(KThreeVector& p0) const
    {
        p0 = fP0;
    }
    void GetP1(KThreeVector& p1) const
    {
        p1 = fP1;
    }

  protected:
    double ClosestApproach(const KThreeVector& aStart, const KThreeVector& anEnd) const;

    KThreeVector fP0;
    KThreeVector fP1;
    double fDiameter;
};
}  // namespace KGeoBag

#endif /* KGMESHTRIANGLE_DEF */
