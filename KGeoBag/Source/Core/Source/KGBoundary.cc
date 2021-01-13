#include "KGBoundary.hh"

namespace KGeoBag
{
KGBoundary::KGBoundary() : fInitialized(false) {}
KGBoundary::KGBoundary(const KGBoundary&) = default;
KGBoundary::~KGBoundary() = default;

void KGBoundary::Accept(KGVisitor* /*aVisitor*/) {}
}  // namespace KGeoBag
