#ifndef Kassiopeia_KSSpace_h_
#define Kassiopeia_KSSpace_h_

#include "KSComponentTemplate.h"
#include "KThreeVector.hh"

#include <set>

namespace Kassiopeia
{

class KSSurface;
class KSSide;

class KSSpace : public KSComponentTemplate<KSSpace>
{
  public:
    friend class KSSurface;
    friend class KSSide;

  public:
    KSSpace();
    ~KSSpace() override;

  public:
    virtual void Enter() const = 0;
    virtual void Exit() const = 0;

    virtual bool Outside(const KGeoBag::KThreeVector& aPoint) const = 0;
    virtual KGeoBag::KThreeVector Point(const KGeoBag::KThreeVector& aPoint) const = 0;
    virtual KGeoBag::KThreeVector Normal(const KGeoBag::KThreeVector& aPoint) const = 0;

    const KSSpace* GetParent() const;
    KSSpace* GetParent();
    void SetParent(KSSpace* aParent);

    int GetSpaceCount() const;
    const KSSpace* GetSpace(int anIndex) const;
    KSSpace* GetSpace(int anIndex);
    void AddSpace(KSSpace* aSpace);
    void RemoveSpace(KSSpace* aSpace);

    int GetSurfaceCount() const;
    const KSSurface* GetSurface(int anIndex) const;
    KSSurface* GetSurface(int anIndex);
    void AddSurface(KSSurface* aSurface);
    void RemoveSurface(KSSurface* aSurface);

    int GetSideCount() const;
    void AddSide(KSSide* aSide);
    KSSide* GetSide(int anIndex);
    const KSSide* GetSide(int anIndex) const;
    void RemoveSide(KSSide* aSide);

  protected:
    KSSpace* fParent;
    std::vector<KSSpace*> fSpaces;
    std::vector<KSSurface*> fSurfaces;
    std::vector<KSSide*> fSides;
};

}  // namespace Kassiopeia

#endif
