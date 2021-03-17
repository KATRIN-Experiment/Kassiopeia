#ifndef KGMESHWIRE_DEF
#define KGMESHWIRE_DEF

#include "KGMeshElement.hh"
#include "KThreeVector.hh"

namespace KGeoBag
{
class KGMeshWire : public KGMeshElement
{
  public:
    KGMeshWire(const KGeoBag::KThreeVector& p0, const KGeoBag::KThreeVector& p1, const double& diameter);
    ~KGMeshWire() override;

    static std::string Name()
    {
        return "mesh_wire";
    }

    double Area() const override;
    double Aspect() const override;
    void Transform(const KTransformation& transform) override;

    double NearestDistance(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector NearestPoint(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector NearestNormal(const KGeoBag::KThreeVector& aPoint) const override;
    bool NearestIntersection(const KGeoBag::KThreeVector& aStart, const KGeoBag::KThreeVector& anEnd,
                             KGeoBag::KThreeVector& anIntersection) const override;

    KGPointCloud<KGMESH_DIM> GetPointCloud() const override;
    unsigned int GetNumberOfEdges() const override
    {
        return 1;
    };
    void GetEdge(KThreeVector& start, KGeoBag::KThreeVector& end, unsigned int /*index*/) const override;

    const KGeoBag::KThreeVector& GetP0() const
    {
        return fP0;
    }
    const KGeoBag::KThreeVector& GetP1() const
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
    double ClosestApproach(const KGeoBag::KThreeVector& aStart, const KGeoBag::KThreeVector& anEnd) const;

    KGeoBag::KThreeVector fP0;
    KGeoBag::KThreeVector fP1;
    double fDiameter;
};
}  // namespace KGeoBag

#endif /* KGMESHTRIANGLE_DEF */
