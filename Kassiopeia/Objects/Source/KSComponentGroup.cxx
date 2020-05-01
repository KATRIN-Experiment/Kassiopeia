#include "KSComponentGroup.h"

using namespace std;

namespace Kassiopeia
{

KSComponentGroup::KSComponentGroup() : KSComponent(), fComponents()
{
    Set(this);
}
KSComponentGroup::KSComponentGroup(const KSComponentGroup& aCopy) : KSComponent(aCopy), fComponents(aCopy.fComponents)
{
    for (auto tIt = fComponents.begin(); tIt != fComponents.end(); tIt++) {
        (*tIt) = (*tIt)->Clone();
    }
    Set(this);
}
KSComponentGroup::~KSComponentGroup()
{
    for (auto tIt = fComponents.begin(); tIt != fComponents.end(); tIt++) {
        delete (*tIt);
    }
}

KSComponentGroup* KSComponentGroup::Clone() const
{
    return new KSComponentGroup(*this);
}
KSComponent* KSComponentGroup::Component(const string& /*aField*/)
{
    return nullptr;
}
KSCommand* KSComponentGroup::Command(const string& /*aField*/, KSComponent* /*aChild*/)
{
    return nullptr;
}

KSComponent* KSComponentGroup::ComponentAt(unsigned int anIndex)
{
    return fComponents.at(anIndex);
}
const KSComponent* KSComponentGroup::ComponentAt(unsigned int anIndex) const
{
    return fComponents.at(anIndex);
}
unsigned int KSComponentGroup::ComponentCount() const
{
    return fComponents.size();
}

void KSComponentGroup::AddComponent(KSComponent* anComponent)
{
    if (anComponent->State() != fState) {
        objctmsg(eError) << "tried to add object <" << anComponent->GetName() << "> in state <" << anComponent->State()
                         << "> to group <" << this->GetName() << "> in state <" << this->State() << ">" << eom;
        return;
    }
    fComponents.push_back(anComponent);
    return;
}
void KSComponentGroup::RemoveComponent(KSComponent* anComponent)
{
    for (auto tIt = fComponents.begin(); tIt != fComponents.end(); tIt++) {
        if ((*tIt) == anComponent) {
            fComponents.erase(tIt);
            return;
        }
    }
    objctmsg(eError) << "cannot remove object <" << anComponent->GetName() << "> from group <" << this->GetName() << ">"
                     << eom;
    return;
}

}  // namespace Kassiopeia
