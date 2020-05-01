#include "KGBoundary.hh"

namespace KGeoBag
{
KGBoundary::KGBoundary() : fInitialized(false) {}
KGBoundary::KGBoundary(const KGBoundary& aCopy) : KTagged(), fInitialized(aCopy.fInitialized) {}
KGBoundary::~KGBoundary() {}

void KGBoundary::Accept(KGVisitor* /*aVisitor*/) {}
}  // namespace KGeoBag
