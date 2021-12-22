#include "KGBoxSpace.hh"

using katrin::KTwoVector;

namespace KGeoBag
{

KGBoxSpace::Visitor::Visitor() = default;
KGBoxSpace::Visitor::~Visitor() = default;

KGBoxSpace::KGBoxSpace() :
    fXA(0.),
    fXB(0.),
    fXMeshCount(2),
    fXMeshPower(1.),
    fYA(0.),
    fYB(0.),
    fYMeshCount(2),
    fYMeshPower(1.),
    fZA(0.),
    fZB(0.),
    fZMeshCount(2),
    fZMeshPower(1.)
{}
KGBoxSpace::~KGBoxSpace() = default;

void KGBoxSpace::XA(const double& aXA)
{
    fXA = aXA;
    return;
}
void KGBoxSpace::XB(const double& aXB)
{
    fXB = aXB;
    return;
}
void KGBoxSpace::XMeshCount(const unsigned int& aXMeshCount)
{
    fXMeshCount = aXMeshCount;
    return;
}
void KGBoxSpace::XMeshPower(const double& aXMeshPower)
{
    fXMeshPower = aXMeshPower;
    return;
}

void KGBoxSpace::YA(const double& aYA)
{
    fYA = aYA;
    return;
}
void KGBoxSpace::YB(const double& aYB)
{
    fYB = aYB;
    return;
}
void KGBoxSpace::YMeshCount(const unsigned int& aYMeshCount)
{
    fYMeshCount = aYMeshCount;
    return;
}
void KGBoxSpace::YMeshPower(const double& aYMeshPower)
{
    fYMeshPower = aYMeshPower;
    return;
}

void KGBoxSpace::ZA(const double& aZA)
{
    fZA = aZA;
    return;
}
void KGBoxSpace::ZB(const double& aZB)
{
    fZB = aZB;
    return;
}
void KGBoxSpace::ZMeshCount(const unsigned int& aZMeshCount)
{
    fZMeshCount = aZMeshCount;
    return;
}
void KGBoxSpace::ZMeshPower(const double& aZMeshPower)
{
    fZMeshPower = aZMeshPower;
    return;
}

const double& KGBoxSpace::XA() const
{
    return fXA;
}
const double& KGBoxSpace::XB() const
{
    return fXB;
}
const unsigned int& KGBoxSpace::XMeshCount() const
{
    return fXMeshCount;
}
const double& KGBoxSpace::XMeshPower() const
{
    return fXMeshPower;
}

const double& KGBoxSpace::YA() const
{
    return fYA;
}
const double& KGBoxSpace::YB() const
{
    return fYB;
}
const unsigned int& KGBoxSpace::YMeshCount() const
{
    return fYMeshCount;
}
const double& KGBoxSpace::YMeshPower() const
{
    return fYMeshPower;
}

const double& KGBoxSpace::ZA() const
{
    return fZA;
}
const double& KGBoxSpace::ZB() const
{
    return fZB;
}
const unsigned int& KGBoxSpace::ZMeshCount() const
{
    return fZMeshCount;
}
const double& KGBoxSpace::ZMeshPower() const
{
    return fZMeshPower;
}

void KGBoxSpace::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const
{
    double tXMin = fXA > fXB ? fXB : fXA;
    double tXMax = fXA > fXB ? fXA : fXB;

    double tYMin = fYA > fYB ? fYB : fYA;
    double tYMax = fYA > fYB ? fYA : fYB;

    fZMin = fZA > fZB ? fZB : fZA;
    fZMax = fZA > fZB ? fZA : fZB;
    fExtrudedMeshCount = fZMeshCount;
    fExtrudedMeshPower = fZMeshPower;

    fPath->StartPoint(KTwoVector(tXMax, tYMax));
    fPath->NextLine(KTwoVector(tXMin, tYMax), fXMeshCount, fXMeshPower);
    fPath->NextLine(KTwoVector(tXMin, tYMin), fYMeshCount, fYMeshPower);
    fPath->NextLine(KTwoVector(tXMax, tYMin), fXMeshCount, fXMeshPower);
    fPath->LastLine(fYMeshCount, fYMeshPower);

    KGExtrudedPolyLoopSpace::VolumeInitialize(aBoundaryContainer);

    return;
}
void KGBoxSpace::VolumeAccept(KGVisitor* aVisitor)
{
    auto* tBoxSpaceVisitor = dynamic_cast<KGBoxSpace::Visitor*>(aVisitor);
    if (tBoxSpaceVisitor != nullptr) {
        tBoxSpaceVisitor->VisitBoxSpace(this);
        return;
    }
    KGExtrudedPolyLoopSpace::VolumeAccept(aVisitor);
    return;
}

}  // namespace KGeoBag
