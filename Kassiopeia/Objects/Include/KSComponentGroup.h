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
    using ComponentIt = ComponentVector::iterator;
    using ComponentCIt = ComponentVector::const_iterator;

    ComponentVector fComponents;
};

}  // namespace Kassiopeia

#endif
