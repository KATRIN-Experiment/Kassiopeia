#ifndef KGRECTANGLE_H_
#define KGRECTANGLE_H_

#include "KGArea.hh"

#include "KThreeVector.hh"

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
    KGRectangle(const double& a, const double& b, const katrin::KThreeVector& p0, const katrin::KThreeVector& n1,
                const katrin::KThreeVector& n2);

    KGRectangle(const katrin::KThreeVector& p0, const katrin::KThreeVector& p1, const katrin::KThreeVector& p2,
                const katrin::KThreeVector& p3);

    ~KGRectangle() override = default;

    void AreaInitialize() const override {}
    void AreaAccept(KGVisitor* aVisitor) override;
    bool AreaAbove(const katrin::KThreeVector& aPoint) const override;
    katrin::KThreeVector AreaPoint(const katrin::KThreeVector& aPoint) const override;
    katrin::KThreeVector AreaNormal(const katrin::KThreeVector& aPoint) const override;

    void SetA(double d)
    {
        fA = d;
    }
    void SetB(double d)
    {
        fB = d;
    }
    void SetP0(const katrin::KThreeVector& p)
    {
        fP0 = p;
    }
    void SetN1(const katrin::KThreeVector& d)
    {
        fN1 = d.Unit();
    }
    void SetN2(const katrin::KThreeVector& d)
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
        return fP0 + fN1 * fA + fN2 * fB;
    }
    const katrin::KThreeVector GetP3() const
    {
        return fP0 + fN2 * fB;
    }

    virtual bool ContainsPoint(const katrin::KThreeVector& aPoint) const;
    double DistanceTo(const katrin::KThreeVector& aPoint, katrin::KThreeVector& nearestPoint);

  protected:
    static bool SameSide(const katrin::KThreeVector& point, const katrin::KThreeVector& A,
                       const katrin::KThreeVector& B, const katrin::KThreeVector& C);

  private:
    double fA;
    double fB;
    katrin::KThreeVector fP0;
    katrin::KThreeVector fN1;
    katrin::KThreeVector fN2;
};
}  // namespace KGeoBag

#endif
