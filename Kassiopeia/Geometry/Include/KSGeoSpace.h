#ifndef Kassiopeia_KSGeoSpace_h_
#define Kassiopeia_KSGeoSpace_h_

#include "KGCore.hh"
#include "KSSpace.h"

namespace Kassiopeia
{

class KSGeoSurface;
class KSGeoSide;

class KSGeoSpace : public KSComponentTemplate<KSGeoSpace, KSSpace>
{

  public:
    KSGeoSpace();
    KSGeoSpace(const KSGeoSpace& aCopy);
    KSGeoSpace* Clone() const override;
    ~KSGeoSpace() override;

  public:
    void Enter() const override;
    void Exit() const override;

    bool Outside(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector Point(const KGeoBag::KThreeVector& aPoint) const override;
    KGeoBag::KThreeVector Normal(const KGeoBag::KThreeVector& aPoint) const override;

  public:
    void AddContent(KGeoBag::KGSpace* aSpace);
    void RemoveContent(KGeoBag::KGSpace* aSpace);
    std::vector<KGeoBag::KGSpace*> GetContent();

    void AddCommand(KSCommand* anCommand);
    void RemoveCommand(KSCommand* anCommand);

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  private:
    mutable std::vector<KGeoBag::KGSpace*> fContents;
    mutable std::vector<KSCommand*> fCommands;
};

}  // namespace Kassiopeia

#endif
