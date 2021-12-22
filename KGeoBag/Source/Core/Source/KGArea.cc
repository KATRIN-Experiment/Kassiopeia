#include "KGArea.hh"

#include "KThreeVector.hh"
using katrin::KThreeVector;

namespace KGeoBag
{
KGArea::KGArea() : fInitialized(false) {}
KGArea::KGArea(const KGArea&) = default;
KGArea::~KGArea() = default;

void KGArea::Accept(KGVisitor* aVisitor)
{
    Check();
    AreaAccept(aVisitor);
}

bool KGArea::Above(const KThreeVector& aPoint) const
{
    Check();
    return AreaAbove(aPoint);
}
KThreeVector KGArea::Point(const KThreeVector& aPoint) const
{
    Check();
    return AreaPoint(aPoint);
}
KThreeVector KGArea::Normal(const KThreeVector& aNormal) const
{
    Check();
    return AreaNormal(aNormal);
}

void KGArea::Check() const
{
    if (fInitialized == false) {
        AreaInitialize();
        fInitialized = true;
    }
    return;
}

void KGArea::AreaAccept(KGVisitor* aVisitor)
{
    auto* tAreaVisitor = dynamic_cast<KGArea::Visitor*>(aVisitor);
    if (tAreaVisitor != nullptr) {
        tAreaVisitor->VisitArea(this);
    }
}

}  // namespace KGeoBag
