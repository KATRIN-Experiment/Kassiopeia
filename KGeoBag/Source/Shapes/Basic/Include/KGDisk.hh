#ifndef KGDISK_H_
#define KGDISK_H_

#include "KGArea.hh"

namespace KGeoBag
{
class KGDisk : public KGArea
{
  public:
    class Visitor
    {
      public:
        Visitor() {}
        virtual ~Visitor() {}

        virtual void Visit(KGDisk*) = 0;
    };

    KGDisk() {}
    KGDisk(const KThreeVector& p0, const KThreeVector& normal, double radius);

    ~KGDisk() override {}


    void AreaInitialize() const override {}
    void AreaAccept(KGVisitor* aVisitor) override;
    bool AreaAbove(const KThreeVector& aPoint) const override;
    KThreeVector AreaPoint(const KThreeVector& aPoint) const override;
    KThreeVector AreaNormal(const KThreeVector& aPoint) const override;

    void SetP0(const KThreeVector& p)
    {
        fP0 = p;
    }
    void SetNormal(const KThreeVector& n)
    {
        fNormal = n.Unit();
    }
    void SetRadius(double d)
    {
        fRadius = d;
    }

    const KThreeVector& GetP0() const
    {
        return fP0;
    }
    const KThreeVector& GetNormal() const
    {
        return fNormal;
    }
    double GetRadius() const
    {
        return fRadius;
    }

  private:
    KThreeVector fP0;
    KThreeVector fNormal;
    double fRadius;
};
}  // namespace KGeoBag

#endif
