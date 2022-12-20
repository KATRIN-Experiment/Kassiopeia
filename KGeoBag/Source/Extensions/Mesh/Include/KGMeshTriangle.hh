#ifndef KGMESHTRIANGLE_DEF
#define KGMESHTRIANGLE_DEF

#include "KGMeshElement.hh"
#include "KGTriangle.hh"

#include "KThreeVector.hh"

namespace KGeoBag
{
class KGMeshTriangle : public KGMeshElement
{
  public:
    KGMeshTriangle(const double& a, const double& b, const katrin::KThreeVector& p0, const katrin::KThreeVector& n1,
                   const katrin::KThreeVector& n2);
    KGMeshTriangle(const katrin::KThreeVector& p0, const katrin::KThreeVector& p1, const katrin::KThreeVector& p2);
    KGMeshTriangle(const KGTriangle& t);
    KGMeshTriangle(const KGMeshTriangle& t);
    ~KGMeshTriangle() override;

    static std::string Name()
    {
        return "mesh_triangle";
    }

    double Area() const override;
    double Aspect() const override;
    katrin::KThreeVector Centroid() const override;
    void Transform(const katrin::KTransformation& transform) override;

    virtual double NearestDistance(const katrin::KThreeVector& aPoint) const override;
    virtual katrin::KThreeVector NearestPoint(const katrin::KThreeVector& aPoint) const override;
    virtual katrin::KThreeVector NearestNormal(const katrin::KThreeVector& aPoint) const override;
    virtual bool NearestIntersection(const katrin::KThreeVector& aStart, const katrin::KThreeVector& anEnd,
                             katrin::KThreeVector& anIntersection) const override;

    KGPointCloud<KGMESH_DIM> GetPointCloud() const override;
    unsigned int GetNumberOfEdges() const override
    {
        return 3;
    };
    void GetEdge(katrin::KThreeVector& start, katrin::KThreeVector& end, unsigned int index) const override;

    //assignment
    inline KGMeshTriangle& operator=(const KGMeshTriangle& t)
    {
        if (&t != this) {
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
    const katrin::KThreeVector& GetP0() const
    {
        return fP0;
    }
    const katrin::KThreeVector& GetN1() const
    {
        return fN1;
    }
    const katrin::KThreeVector& GetN2() const
    {
        return fN2;
    }
    const katrin::KThreeVector GetN3() const
    {
        return fN1.Cross(fN2);
    }
    const katrin::KThreeVector GetP1() const
    {
        return fP0 + fN1 * fA;
    }
    const katrin::KThreeVector GetP2() const
    {
        return fP0 + fN2 * fB;
    }
    void GetP0(katrin::KThreeVector& p0) const
    {
        p0 = fP0;
    }
    void GetN1(katrin::KThreeVector& n1) const
    {
        n1 = fN1;
    }
    void GetN2(katrin::KThreeVector& n2) const
    {
        n2 = fN2;
    }
    void GetN3(katrin::KThreeVector& n3) const
    {
        n3 = fN1.Cross(fN2);
    }
    void GetP1(katrin::KThreeVector& p1) const
    {
        p1 = fP0 + fN1 * fA;
    }
    void GetP2(katrin::KThreeVector& p2) const
    {
        p2 = fP0 + fN2 * fB;
    }

  protected:
    static bool SameSide(const katrin::KThreeVector& point, const katrin::KThreeVector& A,
                       const katrin::KThreeVector& B, const katrin::KThreeVector& C);

    static katrin::KThreeVector NearestPointOnLineSegment(const katrin::KThreeVector& a,
                                                         const katrin::KThreeVector& b,
                                                         const katrin::KThreeVector& point);
    double fA;
    double fB;
    katrin::KThreeVector fP0;
    katrin::KThreeVector fN1;
    katrin::KThreeVector fN2;
};

}  // namespace KGeoBag

#endif /* KGMESHTRIANGLE_DEF */
