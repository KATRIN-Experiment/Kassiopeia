#include "KSSide.h"

#include "KSSpace.h"

namespace Kassiopeia
{

KSSide::KSSide() : fInsideParent(nullptr), fOutsideParent(nullptr) {}
KSSide::~KSSide() {}

const KSSpace* KSSide::GetOutsideParent() const
{
    return fOutsideParent;
}
KSSpace* KSSide::GetOutsideParent()
{
    return fOutsideParent;
}
const KSSpace* KSSide::GetInsideParent() const
{
    return fInsideParent;
}
KSSpace* KSSide::GetInsideParent()
{
    return fInsideParent;
}
void KSSide::SetParent(KSSpace* aParent)
{
    for (auto tSideIt = aParent->fSides.begin(); tSideIt != aParent->fSides.end(); tSideIt++) {
        if ((*tSideIt) == this) {
            aParent->fSides.erase(tSideIt);
            break;
        }
    }

    aParent->fSides.push_back(this);

    this->fInsideParent = aParent;

    this->fOutsideParent = aParent->fParent;

    return;
}

}  // namespace Kassiopeia
