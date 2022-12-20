#ifndef Kassiopeia_KSSide_h_
#define Kassiopeia_KSSide_h_

#include "KSComponentTemplate.h"

#include "KThreeVector.hh"

namespace Kassiopeia
{

class KSSpace;

class KSSide : public KSComponentTemplate<KSSide>
{
  public:
    friend class KSSpace;

  public:
    KSSide();
    ~KSSide() override;

  public:
    virtual void On() const = 0;
    virtual void Off() const = 0;

    virtual katrin::KThreeVector Point(const katrin::KThreeVector& aPoint) const = 0;
    virtual katrin::KThreeVector Normal(const katrin::KThreeVector& aPoint) const = 0;

    const KSSpace* GetOutsideParent() const;
    KSSpace* GetOutsideParent();
    const KSSpace* GetInsideParent() const;
    KSSpace* GetInsideParent();
    void SetParent(KSSpace* aSpace);

  protected:
    KSSpace* fInsideParent;
    KSSpace* fOutsideParent;
};

}  // namespace Kassiopeia

#endif
