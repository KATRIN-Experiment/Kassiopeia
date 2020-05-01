#include "KGExtensible.hh"

namespace KGeoBag
{

KGExtensible::KGExtensible() : fNext(NULL) {}
KGExtensible::KGExtensible(const KGExtensible& aCopy) : fNext(aCopy.fNext->Clone()) {}
KGExtensible& KGExtensible::operator=(const KGExtensible& aCopy)
{
    fNext = aCopy.fNext->Clone();
    return *this;
}
KGExtensible::~KGExtensible()
{
    delete fNext;
}

}  // namespace KGeoBag
