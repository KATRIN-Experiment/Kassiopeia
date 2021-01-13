#ifndef KGBOX_H_
#define KGBOX_H_

#include "KGArea.hh"

namespace KGeoBag
{
class KGBox : public KGArea
{
  public:
    class Visitor
    {
      public:
        Visitor() = default;
        virtual ~Visitor() = default;

        virtual void VisitBox(KGBox*) = 0;
    };

    KGBox();
    KGBox(double x0, double x1, double y0, double y1, double z0, double z1);
    KGBox(const KGeoBag::KThreeVector& p0, const KGeoBag::KThreeVector& p1);

    ~KGBox() override = default;

    void AreaInitialize() const override {}
    void AreaAccept(KGVisitor* aVisitor) override;
    bool AreaAbove(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector AreaPoint(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector AreaNormal(const KGeoBag::KThreeVector& aPoint) const override;

    void SetX0(double d)
    {
        fP0[0] = d;
    }
    void SetY0(double d)
    {
        fP0[1] = d;
    }
    void SetZ0(double d)
    {
        fP0[2] = d;
    }
    void SetX1(double d)
    {
        fP1[0] = d;
    }
    void SetY1(double d)
    {
        fP1[1] = d;
    }
    void SetZ1(double d)
    {
        fP1[2] = d;
    }

    void SetP0(const KGeoBag::KThreeVector& p)
    {
        fP0 = p;
    }
    void SetP1(const KGeoBag::KThreeVector& p)
    {
        fP1 = p;
    }

    double SetX0() const
    {
        return fP0[0];
    }
    double SetY0() const
    {
        return fP0[1];
    }
    double SetZ0() const
    {
        return fP0[2];
    }
    double SetX1() const
    {
        return fP1[0];
    }
    double SetY1() const
    {
        return fP1[1];
    }
    double SetZ1() const
    {
        return fP1[2];
    }

    const KGeoBag::KThreeVector& GetP0() const
    {
        return fP0;
    }
    const KGeoBag::KThreeVector& GetP1() const
    {
        return fP1;
    }

    void SetXMeshCount(unsigned int i)
    {
        fMeshCount[0] = i;
    }
    void SetYMeshCount(unsigned int i)
    {
        fMeshCount[1] = i;
    }
    void SetZMeshCount(unsigned int i)
    {
        fMeshCount[2] = i;
    }

    unsigned int GetXMeshCount() const
    {
        return fMeshCount[0];
    }
    unsigned int GetYMeshCount() const
    {
        return fMeshCount[1];
    }
    unsigned int GetZMeshCount() const
    {
        return fMeshCount[2];
    }

    unsigned int GetMeshCount(unsigned int i) const
    {
        return (i < 3 ? fMeshCount[i] : 0);
    }

    void SetXMeshPower(double d)
    {
        fMeshPower[0] = d;
    }
    void SetYMeshPower(double d)
    {
        fMeshPower[1] = d;
    }
    void SetZMeshPower(double d)
    {
        fMeshPower[2] = d;
    }

    double GetXMeshPower() const
    {
        return fMeshPower[0];
    }
    double GetYMeshPower() const
    {
        return fMeshPower[1];
    }
    double GetZMeshPower() const
    {
        return fMeshPower[2];
    }

    double GetMeshPower(unsigned int i) const
    {
        return (i < 3 ? fMeshPower[i] : 0.);
    }

  private:
    KGeoBag::KThreeVector fP0;
    KGeoBag::KThreeVector fP1;

    unsigned int fMeshCount[3];
    double fMeshPower[3];
};
}  // namespace KGeoBag

#endif
