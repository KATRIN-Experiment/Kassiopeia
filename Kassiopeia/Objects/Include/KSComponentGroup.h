#ifndef Kassiopeia_KSComponentGroup_h_
#define Kassiopeia_KSComponentGroup_h_

#include "KSComponent.h"

namespace Kassiopeia
{

class KSComponentGroup : public KSComponent
{
  public:
    KSComponentGroup();
    KSComponentGroup(const KSComponentGroup& aCopy);
    ~KSComponentGroup() override;

  public:
    KSComponentGroup* Clone() const override;
    KSComponent* Component(const std::string& aField) override;
    KSCommand* Command(const std::string& aField, KSComponent* aChild) override;

  public:
    void AddComponent(KSComponent* aComponent);
    void RemoveComponent(KSComponent* aComponent);

    KSComponent* ComponentAt(unsigned int anIndex);
    const KSComponent* ComponentAt(unsigned int anIndex) const;
    unsigned int ComponentCount() const;

  private:
    typedef std::vector<KSComponent*> ComponentVector;
    typedef ComponentVector::iterator ComponentIt;
    typedef ComponentVector::const_iterator ComponentCIt;

    ComponentVector fComponents;
};

}  // namespace Kassiopeia

#endif
