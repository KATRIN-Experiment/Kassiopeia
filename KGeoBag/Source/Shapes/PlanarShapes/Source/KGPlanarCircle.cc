#include "KGPlanarCircle.hh"

#include "KConst.h"
#include "KGShapeMessage.hh"

namespace KGeoBag
{

KGPlanarCircle::KGPlanarCircle() :
    fCentroid(0., 0.),
    fRadius(0.),
    fMeshCount(32),
    fLength(0.),
    fAnchor(0., 0.),
    fInitialized(false)
{}
KGPlanarCircle::KGPlanarCircle(const KGPlanarCircle& aCopy) :
    fCentroid(aCopy.fCentroid),
    fRadius(aCopy.fRadius),
    fMeshCount(aCopy.fMeshCount),
    fLength(aCopy.fLength),
    fAnchor(aCopy.fAnchor),
    fInitialized(aCopy.fInitialized)
{}
KGPlanarCircle::KGPlanarCircle(const KTwoVector& aCentroid, const double& aRadius, const unsigned int aCount) :
    fCentroid(aCentroid),
    fRadius(aRadius),
    fMeshCount(aCount),
    fLength(0.),
    fAnchor(0., 0.),
    fInitialized(false)
{}
KGPlanarCircle::KGPlanarCircle(const double& anX, const double& aY, const double& aRadius, const unsigned int aCount) :
    fCentroid(anX, aY),
    fRadius(aRadius),
    fMeshCount(aCount),
    fLength(0.),
    fAnchor(0., 0.),
    fInitialized(false)
{}
KGPlanarCircle::~KGPlanarCircle()
{
    shapemsg_debug("destroying a planar circle" << eom);
}

KGPlanarCircle* KGPlanarCircle::Clone() const
{
    return new KGPlanarCircle(*this);
}
void KGPlanarCircle::CopyFrom(const KGPlanarCircle& aCopy)
{
    fCentroid = aCopy.fCentroid;
    fRadius = aCopy.fRadius;
    fMeshCount = aCopy.fMeshCount;
    fLength = aCopy.fLength;
    fAnchor = aCopy.fAnchor;
    fInitialized = aCopy.fInitialized;
    return;
}

void KGPlanarCircle::Centroid(const KTwoVector& aStart)
{
    fInitialized = false;
    fCentroid = aStart;
    return;
}
void KGPlanarCircle::X(const double& aValue)
{
    fInitialized = false;
    fCentroid.X() = aValue;
    return;
}
void KGPlanarCircle::Y(const double& aValue)
{
    fInitialized = false;
    fCentroid.Y() = aValue;
    return;
}
void KGPlanarCircle::Radius(const double& aValue)
{
    fInitialized = false;
    fRadius = aValue;
    return;
}
void KGPlanarCircle::MeshCount(const unsigned int& aCount)
{
    fInitialized = false;
    fMeshCount = aCount;
    return;
}

const KTwoVector& KGPlanarCircle::Centroid() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fCentroid;
}
const double& KGPlanarCircle::X() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fCentroid.X();
}
const double& KGPlanarCircle::Y() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fCentroid.Y();
}
const double& KGPlanarCircle::Radius() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fRadius;
}
const unsigned int& KGPlanarCircle::MeshCount() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fMeshCount;
}

const double& KGPlanarCircle::Length() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fLength;
}
const KTwoVector& KGPlanarCircle::Anchor() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fAnchor;
}

KTwoVector KGPlanarCircle::At(const double& aLength) const
{
    double tAngle = 2. * katrin::KConst::Pi() * aLength / fLength;

    if (tAngle < 0.) {
        tAngle = 0.;
    }
    if (tAngle > 2. * katrin::KConst::Pi()) {
        tAngle = 2. * katrin::KConst::Pi();
    }

    return fCentroid + fRadius * cos(tAngle) * KTwoVector::sXUnit + fRadius * sin(tAngle) * KTwoVector::sYUnit;
}
KTwoVector KGPlanarCircle::Point(const KTwoVector& aPoint) const
{
    KTwoVector tLocalPoint = aPoint - fCentroid;

    return fCentroid + fRadius * tLocalPoint.Unit();
}
KTwoVector KGPlanarCircle::Normal(const KTwoVector& aPoint) const
{
    KTwoVector tLocalPoint = aPoint - fCentroid;

    return tLocalPoint.Unit();
}
bool KGPlanarCircle::Above(const KTwoVector& aPoint) const
{
    KTwoVector tLocalPoint = aPoint - fCentroid;

    if (tLocalPoint.MagnitudeSquared() > fRadius * fRadius) {
        return true;
    }
    else {
        return false;
    }
}

void KGPlanarCircle::Initialize() const
{
    shapemsg_debug("initializing a circle" << eom);

    fLength = 2. * katrin::KConst::Pi() * fRadius;
    fAnchor = fCentroid + fRadius * KTwoVector::sXUnit;

    fInitialized = true;

    return;
}

}  // namespace KGeoBag
