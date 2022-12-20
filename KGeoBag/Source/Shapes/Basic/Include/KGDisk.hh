#ifndef KGDISK_H_
#define KGDISK_H_

#include "KGArea.hh"

#include "KThreeVector.hh"

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
    KGDisk(const katrin::KThreeVector& p0, const katrin::KThreeVector& normal, double radius);

    ~KGDisk() override = default;


    void AreaInitialize() const override {}
    void AreaAccept(KGVisitor* aVisitor) override;
    bool AreaAbove(const katrin::KThreeVector& aPoint) const override;
    katrin::KThreeVector AreaPoint(const katrin::KThreeVector& aPoint) const override;
    katrin::KThreeVector AreaNormal(const katrin::KThreeVector& aPoint) const override;

    void SetP0(const katrin::KThreeVector& p)
    {
        fP0 = p;
    }
    void SetNormal(const katrin::KThreeVector& n)
    {
        fNormal = n.Unit();
    }
    void SetRadius(double d)
    {
        fRadius = d;
    }

    const katrin::KThreeVector& GetP0() const
    {
        return fP0;
    }
    const katrin::KThreeVector& GetNormal() const
    {
        return fNormal;
    }
    double GetRadius() const
    {
        return fRadius;
    }

  private:
    katrin::KThreeVector fP0;
    katrin::KThreeVector fNormal;
    double fRadius;
};
}  // namespace KGeoBag

#endif
