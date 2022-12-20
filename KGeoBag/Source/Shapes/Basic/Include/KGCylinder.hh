#ifndef KGCYLINDER_H_
#define KGCYLINDER_H_

#include "KGArea.hh"

#include "KThreeVector.hh"

namespace KGeoBag
{
class KGCylinder : public KGArea
{
  public:
    class Visitor
    {
      public:
        Visitor() = default;
        virtual ~Visitor() = default;

        virtual void VisitCylinder(KGCylinder*) = 0;
    };

    KGCylinder() : fAxialMeshCount(8), fLongitudinalMeshCount(8), fLongitudinalMeshPower(1.) {}
    KGCylinder(const katrin::KThreeVector& p0, const katrin::KThreeVector& p1, double radius);

    ~KGCylinder() override = default;

    void AreaInitialize() const override {}
    void AreaAccept(KGVisitor* aVisitor) override;
    bool AreaAbove(const katrin::KThreeVector& aPoint) const override;
    katrin::KThreeVector AreaPoint(const katrin::KThreeVector& aPoint) const override;
    katrin::KThreeVector AreaNormal(const katrin::KThreeVector& aPoint) const override;

    void SetP0(const katrin::KThreeVector& p)
    {
        fP0 = p;
    }
    void SetP1(const katrin::KThreeVector& p)
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

    const katrin::KThreeVector& GetP0() const
    {
        return fP0;
    }
    const katrin::KThreeVector& GetP1() const
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
    katrin::KThreeVector fP0;
    katrin::KThreeVector fP1;
    double fRadius;

    unsigned int fAxialMeshCount;
    unsigned int fLongitudinalMeshCount;
    double fLongitudinalMeshPower;
};
}  // namespace KGeoBag

#endif
