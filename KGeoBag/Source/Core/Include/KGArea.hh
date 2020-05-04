#ifndef KGAREA_HH_
#define KGAREA_HH_

#include "KGBoundary.hh"
#include "KGCoreMessage.hh"
#include "KGVisitor.hh"
#include "KTagged.h"
#include "KThreeVector.hh"
#include "KTransformation.hh"
#include "KTwoVector.hh"
using katrin::KTagged;

#include "KConst.h"

#include <cmath>

namespace KGeoBag
{
class KGArea : public KGBoundary
{
  public:
    class Visitor
    {
      public:
        Visitor() {}
        virtual ~Visitor() {}
        virtual void VisitArea(KGArea*) = 0;
    };

  public:
    KGArea();
    KGArea(const KGArea& aArea);
    ~KGArea() override;

  public:
    void Accept(KGVisitor* aVisitor);

  protected:
    virtual void AreaAccept(KGVisitor* aVisitor);

  public:
    bool Above(const KThreeVector& aPoint) const;
    KThreeVector Point(const KThreeVector& aPoint) const;
    KThreeVector Normal(const KThreeVector& aPoint) const;

  protected:
    virtual bool AreaAbove(const KThreeVector& aPoint) const = 0;
    virtual KThreeVector AreaPoint(const KThreeVector& aPoint) const = 0;
    virtual KThreeVector AreaNormal(const KThreeVector& aPoint) const = 0;

  protected:
    void Check() const;
    virtual void AreaInitialize() const override = 0;
    mutable bool fInitialized;
};

}  // namespace KGeoBag

#endif
