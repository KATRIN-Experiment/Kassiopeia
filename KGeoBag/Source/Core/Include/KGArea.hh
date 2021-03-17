#ifndef KGAREA_HH_
#define KGAREA_HH_

#include "KConst.h"
#include "KGBoundary.hh"
#include "KGCoreMessage.hh"
#include "KGVisitor.hh"
#include "KTagged.h"
#include "KThreeVector.hh"
#include "KTransformation.hh"
#include "KTwoVector.hh"

#include <cmath>

namespace KGeoBag
{
class KGArea : public KGBoundary
{
  public:
    class Visitor
    {
      public:
        Visitor() = default;
        virtual ~Visitor() = default;
        virtual void VisitArea(KGArea*) = 0;
    };

  public:
    KGArea();
    KGArea(const KGArea& aArea);
    ~KGArea() override;

    static std::string Name()
    {
        return "area";
    }

  public:
    void Accept(KGVisitor* aVisitor);

  protected:
    virtual void AreaAccept(KGVisitor* aVisitor);

  public:
    bool Above(const KGeoBag::KThreeVector& aPoint) const;
    KGeoBag::KThreeVector Point(const KGeoBag::KThreeVector& aPoint) const;
    KGeoBag::KThreeVector Normal(const KGeoBag::KThreeVector& aPoint) const;

  protected:
    virtual bool AreaAbove(const KGeoBag::KThreeVector& aPoint) const = 0;
    virtual KGeoBag::KThreeVector AreaPoint(const KGeoBag::KThreeVector& aPoint) const = 0;
    virtual KGeoBag::KThreeVector AreaNormal(const KGeoBag::KThreeVector& aPoint) const = 0;

  protected:
    void Check() const;
    void AreaInitialize() const override = 0;
    mutable bool fInitialized;
};

}  // namespace KGeoBag

#endif
