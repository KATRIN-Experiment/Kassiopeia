#ifndef KGBOUNDARY_HH_
#define KGBOUNDARY_HH_

#include "KGVisitor.hh"
#include "KTagged.h"
using katrin::KTagged;

namespace KGeoBag
{
class KGBoundary : public KTagged
{
  public:
    class Visitor
    {
      public:
        Visitor() {}
        virtual ~Visitor() {}
    };

  public:
    KGBoundary();
    KGBoundary(const KGBoundary& aBoundary);
    ~KGBoundary() override;

  public:
    void Accept(KGVisitor* aVisitor);

  protected:
    virtual void AreaInitialize() const = 0;
    mutable bool fInitialized;
};

}  // namespace KGeoBag

#endif
