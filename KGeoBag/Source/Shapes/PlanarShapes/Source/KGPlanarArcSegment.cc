#include "KGPlanarArcSegment.hh"

#include "KGShapeMessage.hh"

using katrin::KTwoVector;

namespace KGeoBag
{

KGPlanarArcSegment::KGPlanarArcSegment() :
    fStart(0., 0.),
    fEnd(0., 0.),
    fRadius(0.),
    fRight(true),
    fShort(true),
    fMeshCount(16),
    fLength(0.),
    fAngle(0.),
    fCentroid(0., 0.),
    fOrigin(0., 0.),
    fXUnit(1., 0.),
    fYUnit(0., 1.),
    fInitialized(false)
{}
KGPlanarArcSegment::KGPlanarArcSegment(const KGPlanarArcSegment& aCopy) :
    fStart(aCopy.fStart),
    fEnd(aCopy.fEnd),
    fRadius(aCopy.fRadius),
    fRight(aCopy.fRight),
    fShort(aCopy.fShort),
    fMeshCount(aCopy.fMeshCount),
    fLength(aCopy.fLength),
    fAngle(aCopy.fAngle),
    fCentroid(aCopy.fCentroid),
    fOrigin(aCopy.fOrigin),
    fXUnit(aCopy.fXUnit),
    fYUnit(aCopy.fYUnit),
    fInitialized(aCopy.fInitialized)
{}
KGPlanarArcSegment::KGPlanarArcSegment(const KTwoVector& aStart, const KTwoVector& anEnd, const double& aRadius,
                                       const bool& isRight, const bool& isShort, const unsigned int aCount) :
    fStart(aStart),
    fEnd(anEnd),
    fRadius(aRadius),
    fRight(isRight),
    fShort(isShort),
    fMeshCount(aCount),
    fLength(0.),
    fAngle(0.),
    fCentroid(0., 0.),
    fOrigin(0., 0.),
    fXUnit(1., 0.),
    fYUnit(0., 1.),
    fInitialized(false)
{}
KGPlanarArcSegment::KGPlanarArcSegment(const double& anX1, const double& aY1, const double& anX2, const double& aY2,
                                       const double& aRadius, const bool& isRight, const bool& isShort,
                                       const unsigned int aCount) :
    fStart(anX1, aY1),
    fEnd(anX2, aY2),
    fRadius(aRadius),
    fRight(isRight),
    fShort(isShort),
    fMeshCount(aCount),
    fLength(0.),
    fAngle(0.),
    fCentroid(0., 0.),
    fOrigin(0., 0.),
    fXUnit(1., 0.),
    fYUnit(0., 1.),
    fInitialized(false)
{}
KGPlanarArcSegment::~KGPlanarArcSegment()
{
    shapemsg_debug("destroying a planar arc segment" << eom);
}

KGPlanarArcSegment* KGPlanarArcSegment::Clone() const
{
    return new KGPlanarArcSegment(*this);
}
void KGPlanarArcSegment::CopyFrom(const KGPlanarArcSegment& aCopy)
{
    fStart = aCopy.fStart;
    fEnd = aCopy.fEnd;
    fRadius = aCopy.fRadius;
    fRight = aCopy.fRight;
    fShort = aCopy.fShort;
    fMeshCount = aCopy.fMeshCount;
    fLength = aCopy.fLength;
    fAngle = aCopy.fAngle;
    fCentroid = aCopy.fCentroid;
    fOrigin = aCopy.fOrigin;
    fXUnit = aCopy.fXUnit;
    fYUnit = aCopy.fYUnit;
    fInitialized = aCopy.fInitialized;
    return;
}

void KGPlanarArcSegment::Start(const KTwoVector& aStart)
{
    fInitialized = false;
    fStart = aStart;
    return;
}
void KGPlanarArcSegment::X1(const double& aValue)
{
    fInitialized = false;
    fStart.X() = aValue;
    return;
}
void KGPlanarArcSegment::Y1(const double& aValue)
{
    fInitialized = false;
    fStart.Y() = aValue;
    return;
}
void KGPlanarArcSegment::End(const KTwoVector& anEnd)
{
    fInitialized = false;
    fEnd = anEnd;
    return;
}
void KGPlanarArcSegment::X2(const double& aValue)
{
    fInitialized = false;
    fEnd.X() = aValue;
    return;
}
void KGPlanarArcSegment::Y2(const double& aValue)
{
    fInitialized = false;
    fEnd.Y() = aValue;
    return;
}
void KGPlanarArcSegment::Radius(const double& aValue)
{
    fInitialized = false;
    fRadius = aValue;
    return;
}
void KGPlanarArcSegment::Right(const bool& aValue)
{
    fInitialized = false;
    fRight = aValue;
    return;
}
void KGPlanarArcSegment::Short(const bool& aValue)
{
    fInitialized = false;
    fShort = aValue;
    return;
}
void KGPlanarArcSegment::MeshCount(const unsigned int& aCount)
{
    fInitialized = false;
    fMeshCount = aCount;
    return;
}

const KTwoVector& KGPlanarArcSegment::Start() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fStart;
}
const double& KGPlanarArcSegment::X1() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fStart.X();
}
const double& KGPlanarArcSegment::Y1() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fStart.Y();
}
const KTwoVector& KGPlanarArcSegment::End() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fEnd;
}
const double& KGPlanarArcSegment::X2() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fEnd.X();
}
const double& KGPlanarArcSegment::Y2() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fEnd.Y();
}
const double& KGPlanarArcSegment::Radius() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fRadius;
}
const bool& KGPlanarArcSegment::Right() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fRight;
}
const bool& KGPlanarArcSegment::Short() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fShort;
}
const unsigned int& KGPlanarArcSegment::MeshCount() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fMeshCount;
}

const double& KGPlanarArcSegment::Length() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fLength;
}
const double& KGPlanarArcSegment::Angle() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fAngle;
}
const KTwoVector& KGPlanarArcSegment::Centroid() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fCentroid;
}
const KTwoVector& KGPlanarArcSegment::Origin() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fOrigin;
}
const KTwoVector& KGPlanarArcSegment::XUnit() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fXUnit;
}
const KTwoVector& KGPlanarArcSegment::YUnit() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fYUnit;
}

KTwoVector KGPlanarArcSegment::At(const double& aLength) const
{
    if (fInitialized == false) {
        Initialize();
    }

    double tAngle = (2. * (aLength / fLength) - 1.) * fAngle;

    if (tAngle < -fAngle) {
        tAngle = -fAngle;
    }
    if (tAngle > fAngle) {
        tAngle = fAngle;
    }
    if (fRight == false) {
        tAngle = -tAngle;
    }

    return fOrigin + fRadius * cos(tAngle) * fXUnit + fRadius * sin(tAngle) * fYUnit;
}
KTwoVector KGPlanarArcSegment::Point(const KTwoVector& aPoint) const
{
    if (fInitialized == false) {
        Initialize();
    }

    double tX = fXUnit.X() * (aPoint.X() - fOrigin.X()) + fXUnit.Y() * (aPoint.Y() - fOrigin.Y());
    double tY = fYUnit.X() * (aPoint.X() - fOrigin.X()) + fYUnit.Y() * (aPoint.Y() - fOrigin.Y());
    double tAngle = atan2(tY, tX);

    if (tAngle < -fAngle) {
        tAngle = -fAngle;
    }

    if (tAngle > fAngle) {
        tAngle = fAngle;
    }

    return fOrigin + fRadius * cos(tAngle) * fXUnit + fRadius * sin(tAngle) * fYUnit;
}
KTwoVector KGPlanarArcSegment::Normal(const KTwoVector& aPoint) const
{
    if (fInitialized == false) {
        Initialize();
    }

    double tX = fXUnit.X() * (aPoint.X() - fOrigin.X()) + fXUnit.Y() * (aPoint.Y() - fOrigin.Y());
    double tY = fYUnit.X() * (aPoint.X() - fOrigin.X()) + fYUnit.Y() * (aPoint.Y() - fOrigin.Y());
    double tAngle = atan2(tY, tX);

    if (tAngle < -fAngle) {
        tAngle = -fAngle;
    }

    if (tAngle > fAngle) {
        tAngle = fAngle;
    }

    if (fRight == true) {
        return cos(tAngle) * fXUnit + sin(tAngle) * fYUnit;
    }
    else {
        return -1. * cos(tAngle) * fXUnit + -1. * sin(tAngle) * fYUnit;
    }
}
bool KGPlanarArcSegment::Above(const KTwoVector& aPoint) const
{
    if (fInitialized == false) {
        Initialize();
    }

    double tR = (aPoint - fOrigin).Magnitude();
    double tX = fXUnit.X() * (aPoint.X() - fStart.X()) + fXUnit.Y() * (aPoint.Y() - fStart.Y());

    if (fRight == true) {
        if (tX > 0.) {
            if (tR < fRadius) {
                return false;
            }
            return true;
        }
        return false;
    }
    else {
        if (tX > 0.) {
            if (tR < fRadius) {
                return true;
            }
            return false;
        }
        return true;
    }
}

void KGPlanarArcSegment::Initialize() const
{
    shapemsg_debug("initializing a planar arc segment" << eom);

    KTwoVector tLine = fEnd - fStart;
    KTwoVector tLineUnit = tLine.Unit();
    KTwoVector tLineOrthogonalUnit = tLine.Orthogonal(false).Unit();
    double tLineLength = tLine.Magnitude() / 2.;
    double tLineOrthogonalLength = sqrt(fRadius * fRadius - tLineLength * tLineLength);

    if (fRight == true) {
        if (fShort == true) {
            fXUnit = tLineOrthogonalUnit;
            fYUnit = tLineUnit;
            fOrigin = fStart + tLineLength * tLineUnit - tLineOrthogonalLength * tLineOrthogonalUnit;
            fAngle = atan2(tLineLength, tLineOrthogonalLength);
        }
        if (fShort == false) {
            fXUnit = tLineOrthogonalUnit;
            fYUnit = tLineUnit;
            fOrigin = fStart + tLineLength * tLineUnit + tLineOrthogonalLength * tLineOrthogonalUnit;
            fAngle = atan2(tLineLength, -tLineOrthogonalLength);
        }
    }
    if (fRight == false) {
        if (fShort == true) {
            fXUnit = -1. * tLineOrthogonalUnit;
            fYUnit = -1. * tLineUnit;
            fOrigin = fStart + tLineLength * tLineUnit + tLineOrthogonalLength * tLineOrthogonalUnit;
            fAngle = atan2(tLineLength, tLineOrthogonalLength);
        }
        if (fShort == false) {
            fXUnit = -1. * tLineOrthogonalUnit;
            fYUnit = -1. * tLineUnit;
            fOrigin = fStart + tLineLength * tLineUnit - tLineOrthogonalLength * tLineOrthogonalUnit;
            fAngle = atan2(tLineLength, -tLineOrthogonalLength);
        }
    }

    fLength = 2. * fRadius * fAngle;
    fCentroid = fOrigin + (fRadius * sin(fAngle) / fAngle) * fXUnit;

    shapemsg_debug("  x unit is <" << fXUnit << ">" << eom);

    fInitialized = true;

    return;
}

}  // namespace KGeoBag
