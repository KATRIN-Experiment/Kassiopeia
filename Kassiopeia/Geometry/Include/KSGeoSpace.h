#ifndef Kassiopeia_KSGeoSpace_h_
#define Kassiopeia_KSGeoSpace_h_

#include "KGCore.hh"
#include "KSSpace.h"
using namespace KGeoBag;

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

    bool Outside(const KThreeVector& aPoint) const override;
    KThreeVector Point(const KThreeVector& aPoint) const override;
    KThreeVector Normal(const KThreeVector& aPoint) const override;

  public:
    void AddContent(KGSpace* aSpace);
    void RemoveContent(KGSpace* aSpace);
    std::vector<KGSpace*> GetContent();

    void AddCommand(KSCommand* anCommand);
    void RemoveCommand(KSCommand* anCommand);

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  private:
    mutable std::vector<KGSpace*> fContents;
    mutable std::vector<KSCommand*> fCommands;
};

}  // namespace Kassiopeia

#endif
