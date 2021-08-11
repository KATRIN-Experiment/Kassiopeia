#ifndef KGTRIANGLE_H_
#define KGTRIANGLE_H_

#include "KGArea.hh"

namespace KGeoBag
{
class KGTriangle : public KGArea
{
  public:
    class Visitor
    {
      public:
        Visitor() = default;
        virtual ~Visitor() = default;

        virtual void Visit(KGTriangle*) = 0;
    };

    KGTriangle() = default;
    KGTriangle(const double& a, const double& b, const KGeoBag::KThreeVector& p0, const KGeoBag::KThreeVector& n1,
                const KGeoBag::KThreeVector& n2);

    KGTriangle(const KGeoBag::KThreeVector& p0, const KGeoBag::KThreeVector& p1, const KGeoBag::KThreeVector& p2);
    KGTriangle(const KGTriangle&);
    KGTriangle& operator=(const KGTriangle&);

    ~KGTriangle() override = default;

    void AreaInitialize() const override {}
    void AreaAccept(KGVisitor* aVisitor) override;
    bool AreaAbove(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector AreaPoint(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector AreaNormal(const KGeoBag::KThreeVector& aPoint) const override;

    void SetA(double d)
    {
        fA = d;
    }
    void SetB(double d)
    {
        fB = d;
    }
    void SetP0(const KGeoBag::KThreeVector& p)
    {
        fP0 = p;
    }
    void SetN1(const KGeoBag::KThreeVector& d)
    {
        fN1 = d.Unit();
    }
    void SetN2(const KGeoBag::KThreeVector& d)
    {
        fN2 = d.Unit();
    }

    double GetA() const
    {
        return fA;
    }
    double GetB() const
    {
        return fB;
    }
    const KGeoBag::KThreeVector& GetP0() const
    {
        return fP0;
    }
    const KGeoBag::KThreeVector& GetN1() const
    {
        return fN1;
    }
    const KGeoBag::KThreeVector& GetN2() const
    {
        return fN2;
    }
    const KGeoBag::KThreeVector GetN3() const
    {
        return fN1.Cross(fN2);
    }
    const KGeoBag::KThreeVector GetP1() const
    {
        return fP0 + fN1 * fA;
    }
    const KGeoBag::KThreeVector GetP2() const
    {
        return fP0 + fN2 * fB;
    }

    virtual bool ContainsPoint(const KThreeVector& aPoint) const;
    double DistanceTo(const KThreeVector& aPoint, KThreeVector& nearestPoint);

  protected:
    static bool SameSide(const KGeoBag::KThreeVector& point, const KGeoBag::KThreeVector& A,
                       const KGeoBag::KThreeVector& B, const KGeoBag::KThreeVector& C);

  protected:
    double fA;
    double fB;
    KGeoBag::KThreeVector fP0;
    KGeoBag::KThreeVector fN1;
    KGeoBag::KThreeVector fN2;

};
}  // namespace KGeoBag

#endif
