#ifndef KGBOUNDARY_HH_
#define KGBOUNDARY_HH_

#include "KGVisitor.hh"
#include "KTagged.h"

namespace KGeoBag
{
class KGBoundary : public katrin::KTagged
{
  public:
    class Visitor
    {
      public:
        Visitor() = default;
        virtual ~Visitor() = default;
    };

  public:
    KGBoundary();
    KGBoundary(const KGBoundary& aBoundary);
    ~KGBoundary() override;

    static std::string Name()
    {
        return "boundary";
    }

  public:
    void Accept(KGVisitor* aVisitor);

  protected:
    virtual void AreaInitialize() const = 0;
    mutable bool fInitialized;
};

}  // namespace KGeoBag

#endif
