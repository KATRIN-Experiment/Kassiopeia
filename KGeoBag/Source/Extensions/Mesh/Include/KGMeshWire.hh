#ifndef KGMESHWIRE_DEF
#define KGMESHWIRE_DEF

#include "KGMeshElement.hh"

#include "KThreeVector.hh"

namespace KGeoBag
{
class KGMeshWire : public KGMeshElement
{
  public:
    KGMeshWire(const katrin::KThreeVector& p0, const katrin::KThreeVector& p1, const double& diameter);
    ~KGMeshWire() override;

    static std::string Name()
    {
        return "mesh_wire";
    }

    double Area() const override;
    double Aspect() const override;
    katrin::KThreeVector Centroid() const override;
    void Transform(const katrin::KTransformation& transform) override;

    double NearestDistance(const katrin::KThreeVector& aPoint) const override;
    katrin::KThreeVector NearestPoint(const katrin::KThreeVector& aPoint) const override;
    katrin::KThreeVector NearestNormal(const katrin::KThreeVector& aPoint) const override;
    bool NearestIntersection(const katrin::KThreeVector& aStart, const katrin::KThreeVector& anEnd,
                             katrin::KThreeVector& anIntersection) const override;

    KGPointCloud<KGMESH_DIM> GetPointCloud() const override;
    unsigned int GetNumberOfEdges() const override
    {
        return 1;
    };
    void GetEdge(katrin::KThreeVector& start, katrin::KThreeVector& end, unsigned int /*index*/) const override;

    const katrin::KThreeVector& GetP0() const
    {
        return fP0;
    }
    const katrin::KThreeVector& GetP1() const
    {
        return fP1;
    }
    double GetDiameter() const
    {
        return fDiameter;
    }
    void GetP0(katrin::KThreeVector& p0) const
    {
        p0 = fP0;
    }
    void GetP1(katrin::KThreeVector& p1) const
    {
        p1 = fP1;
    }

  protected:
    double ClosestApproach(const katrin::KThreeVector& aStart, const katrin::KThreeVector& anEnd) const;

    katrin::KThreeVector fP0;
    katrin::KThreeVector fP1;
    double fDiameter;
};
}  // namespace KGeoBag

#endif /* KGMESHTRIANGLE_DEF */
