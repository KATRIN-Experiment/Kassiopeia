#ifndef Kassiopeia_KSSurface_h_
#define Kassiopeia_KSSurface_h_

#include "KSComponentTemplate.h"
#include "KThreeVector.hh"

namespace Kassiopeia
{

class KSSpace;

class KSSurface : public KSComponentTemplate<KSSurface>
{
  public:
    friend class KSSpace;

  public:
    KSSurface();
    ~KSSurface() override;

  public:
    virtual void On() const = 0;
    virtual void Off() const = 0;

    virtual KGeoBag::KThreeVector Point(const KGeoBag::KThreeVector& aPoint) const = 0;
    virtual KGeoBag::KThreeVector Normal(const KGeoBag::KThreeVector& aPoint) const = 0;

    const KSSpace* GetParent() const;
    KSSpace* GetParent();
    void SetParent(KSSpace* aSpace);

  protected:
    KSSpace* fParent;
};

}  // namespace Kassiopeia

#endif
