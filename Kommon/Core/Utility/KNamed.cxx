#include "KNamed.h"

namespace katrin
{

KNamed::KNamed() : fName("(anonymous)") {}
KNamed::KNamed(const KNamed& aNamed) : fName(aNamed.fName) {}

bool KNamed::HasName(const std::string& aName) const
{
    return fName == aName;
}
void KNamed::SetName(std::string aName)
{
    fName = std::move(aName);
}
const std::string& KNamed::GetName() const
{
    return fName;
}

void KNamed::Print(std::ostream& output) const
{
    output << "<" << GetName() << ">";
}

}  // namespace katrin
