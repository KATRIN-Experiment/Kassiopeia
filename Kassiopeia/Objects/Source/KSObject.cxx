#include "KSObject.h"

namespace Kassiopeia
{

KSObject::KSObject() : KTagged() {};
KSObject::KSObject(const KSObject& aCopy) : KTagged(aCopy) {}
KSObject::~KSObject() = default;

}  // namespace Kassiopeia
