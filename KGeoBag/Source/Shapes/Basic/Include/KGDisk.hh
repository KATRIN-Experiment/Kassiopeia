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
        Visitor() = default;
        virtual ~Visitor() = default;

        virtual void Visit(KGDisk*) = 0;
    };

    KGDisk() = default;
    KGDisk(const KGeoBag::KThreeVector& p0, const KGeoBag::KThreeVector& normal, double radius);

    ~KGDisk() override = default;


    void AreaInitialize() const override {}
    void AreaAccept(KGVisitor* aVisitor) override;
    bool AreaAbove(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector AreaPoint(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector AreaNormal(const KGeoBag::KThreeVector& aPoint) const override;

    void SetP0(const KGeoBag::KThreeVector& p)
    {
        fP0 = p;
    }
    void SetNormal(const KGeoBag::KThreeVector& n)
    {
        fNormal = n.Unit();
    }
    void SetRadius(double d)
    {
        fRadius = d;
    }

    const KGeoBag::KThreeVector& GetP0() const
    {
        return fP0;
    }
    const KGeoBag::KThreeVector& GetNormal() const
    {
        return fNormal;
    }
    double GetRadius() const
    {
        return fRadius;
    }

  private:
    KGeoBag::KThreeVector fP0;
    KGeoBag::KThreeVector fNormal;
    double fRadius;
};
}  // namespace KGeoBag

#endif
