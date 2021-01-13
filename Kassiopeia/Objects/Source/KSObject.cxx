#include "KSObject.h"

namespace Kassiopeia
{

KSObject::KSObject() : KTagged(), fHolder(nullptr) {}
KSObject::KSObject(const KSObject& aCopy) : KTagged(aCopy), fHolder(nullptr) {}
KSObject::~KSObject() = default;

}  // namespace Kassiopeia
