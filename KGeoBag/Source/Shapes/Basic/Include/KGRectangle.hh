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
        Visitor() {}
        virtual ~Visitor() {}

        virtual void Visit(KGRectangle*) = 0;
    };

    KGRectangle() {}
    KGRectangle(const double& a, const double& b, const KThreeVector& p0, const KThreeVector& n1,
                const KThreeVector& n2);

    KGRectangle(const KThreeVector& p0, const KThreeVector& p1, const KThreeVector& p2, const KThreeVector& p3);

    ~KGRectangle() override {}

    void AreaInitialize() const override {}
    void AreaAccept(KGVisitor* aVisitor) override;
    bool AreaAbove(const KThreeVector& aPoint) const override;
    KThreeVector AreaPoint(const KThreeVector& aPoint) const override;
    KThreeVector AreaNormal(const KThreeVector& aPoint) const override;

    void SetA(double d)
    {
        fA = d;
    }
    void SetB(double d)
    {
        fB = d;
    }
    void SetP0(const KThreeVector& p)
    {
        fP0 = p;
    }
    void SetN1(const KThreeVector& d)
    {
        fN1 = d.Unit();
    }
    void SetN2(const KThreeVector& d)
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

  private:
    double fA;
    double fB;
    KThreeVector fP0;
    KThreeVector fN1;
    KThreeVector fN2;
};
}  // namespace KGeoBag

#endif
