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
    KGDisk(const KThreeVector& p0,
	   const KThreeVector& normal,
	   double radius);

    virtual ~KGDisk() {}


    virtual void AreaInitialize() const {}
    virtual void AreaAccept(KGVisitor* aVisitor);
    virtual bool AreaAbove(const KThreeVector& aPoint) const;
    virtual KThreeVector AreaPoint(const KThreeVector& aPoint) const;
    virtual KThreeVector AreaNormal(const KThreeVector& aPoint) const;

    void SetP0(const KThreeVector& p) { fP0 = p; }
    void SetNormal(const KThreeVector& n) { fNormal = n.Unit(); }
    void SetRadius(double d) { fRadius = d; }

    const KThreeVector& GetP0() const { return fP0; }
    const KThreeVector& GetNormal() const { return fNormal; }
    double GetRadius() const { return fRadius; }

  private:

    KThreeVector fP0;
    KThreeVector fNormal;
    double fRadius;
  };
}

#endif
