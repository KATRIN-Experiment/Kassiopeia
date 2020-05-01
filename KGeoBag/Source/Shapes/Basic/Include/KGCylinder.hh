#ifndef KGCYLINDER_H_
#define KGCYLINDER_H_

#include "KGArea.hh"

namespace KGeoBag
{
class KGCylinder : public KGArea
{
  public:
    class Visitor
    {
      public:
        Visitor() {}
        virtual ~Visitor() {}

        virtual void VisitCylinder(KGCylinder*) = 0;
    };

    KGCylinder() : fAxialMeshCount(8), fLongitudinalMeshCount(8), fLongitudinalMeshPower(1.) {}
    KGCylinder(const KThreeVector& p0, const KThreeVector& p1, double radius);

    ~KGCylinder() override {}

    void AreaInitialize() const override {}
    void AreaAccept(KGVisitor* aVisitor) override;
    bool AreaAbove(const KThreeVector& aPoint) const override;
    KThreeVector AreaPoint(const KThreeVector& aPoint) const override;
    KThreeVector AreaNormal(const KThreeVector& aPoint) const override;

    void SetP0(const KThreeVector& p)
    {
        fP0 = p;
    }
    void SetP1(const KThreeVector& p)
    {
        fP1 = p;
    }
    void SetRadius(double d)
    {
        fRadius = d;
    }
    void SetAxialMeshCount(unsigned int i)
    {
        fAxialMeshCount = i;
    }
    void SetLongitudinalMeshCount(unsigned int i)
    {
        fLongitudinalMeshCount = i;
    }
    void SetLongitudinalMeshPower(double d)
    {
        fLongitudinalMeshPower = d;
    }

    const KThreeVector& GetP0() const
    {
        return fP0;
    }
    const KThreeVector& GetP1() const
    {
        return fP1;
    }
    double GetRadius() const
    {
        return fRadius;
    }
    unsigned int GetAxialMeshCount() const
    {
        return fAxialMeshCount;
    }
    unsigned int GetLongitudinalMeshCount() const
    {
        return fLongitudinalMeshCount;
    }
    double GetLongitudinalMeshPower() const
    {
        return fLongitudinalMeshPower;
    }

  private:
    KThreeVector fP0;
    KThreeVector fP1;
    double fRadius;

    unsigned int fAxialMeshCount;
    unsigned int fLongitudinalMeshCount;
    double fLongitudinalMeshPower;
};
}  // namespace KGeoBag

#endif
