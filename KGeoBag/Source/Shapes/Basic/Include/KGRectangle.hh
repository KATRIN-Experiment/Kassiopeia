#ifndef KGRECTANGLE_H_
#define KGRECTANGLE_H_

#include "KGArea.hh"

namespace KGeoBag
{
class KGRectangle : public KGArea
{
  public:
    class Visitor
    {
      public:
        Visitor() = default;
        virtual ~Visitor() = default;

        virtual void Visit(KGRectangle*) = 0;
    };

    KGRectangle() = default;
    KGRectangle(const double& a, const double& b, const KGeoBag::KThreeVector& p0, const KGeoBag::KThreeVector& n1,
                const KGeoBag::KThreeVector& n2);

    KGRectangle(const KGeoBag::KThreeVector& p0, const KGeoBag::KThreeVector& p1, const KGeoBag::KThreeVector& p2,
                const KGeoBag::KThreeVector& p3);

    ~KGRectangle() override = default;

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
        return fP0 + fN1 * fA + fN2 * fB;
    }
    const KGeoBag::KThreeVector GetP3() const
    {
        return fP0 + fN2 * fB;
    }

  private:
    double fA;
    double fB;
    KGeoBag::KThreeVector fP0;
    KGeoBag::KThreeVector fN1;
    KGeoBag::KThreeVector fN2;
};
}  // namespace KGeoBag

#endif
