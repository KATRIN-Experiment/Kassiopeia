#include "KGPlanarPolyLine.hh"

#include "KGShapeMessage.hh"

namespace KGeoBag
{

KGPlanarPolyLine::KGPlanarPolyLine() : fLength(0.), fCentroid(0., 0.), fStart(0., 0.), fEnd(0., 0.), fInitialized(false)
{}
KGPlanarPolyLine::KGPlanarPolyLine(const KGPlanarPolyLine& aCopy) :
    fLength(aCopy.fLength),
    fCentroid(aCopy.fCentroid),
    fStart(aCopy.fStart),
    fEnd(aCopy.fEnd),
    fInitialized(aCopy.fInitialized)
{
    const KGPlanarOpenPath* tElement;
    const KGPlanarLineSegment* tLineSegment;
    const KGPlanarArcSegment* tArcSegment;
    for (const auto* element : aCopy.fElements) {
        tElement = element;

        tLineSegment = dynamic_cast<const KGPlanarLineSegment*>(tElement);
        if (tLineSegment != nullptr) {
            fElements.push_back(new KGPlanarLineSegment(*tLineSegment));
            continue;
        }

        tArcSegment = dynamic_cast<const KGPlanarArcSegment*>(tElement);
        if (tArcSegment != nullptr) {
            fElements.push_back(new KGPlanarArcSegment(*tArcSegment));
            continue;
        }
    }
}
KGPlanarPolyLine::~KGPlanarPolyLine()
{
    shapemsg_debug("destroying a planar poly line" << eom);

    const KGPlanarOpenPath* tElement;
    for (auto& element : fElements) {
        tElement = element;
        delete tElement;
    }
}

KGPlanarPolyLine* KGPlanarPolyLine::Clone() const
{
    return new KGPlanarPolyLine(*this);
}
void KGPlanarPolyLine::CopyFrom(const KGPlanarPolyLine& aCopy)
{
    fLength = aCopy.fLength;
    fCentroid = aCopy.fCentroid;
    fStart = aCopy.fStart;
    fEnd = aCopy.fEnd;
    fInitialized = aCopy.fInitialized;

    const KGPlanarOpenPath* tElement;
    for (auto& element : fElements) {
        tElement = element;
        delete tElement;
    }
    fElements.clear();

    const KGPlanarLineSegment* tLineSegment;
    const KGPlanarArcSegment* tArcSegment;
    for (const auto* element : aCopy.fElements) {
        tElement = element;

        tLineSegment = dynamic_cast<const KGPlanarLineSegment*>(tElement);
        if (tLineSegment != nullptr) {
            fElements.push_back(new KGPlanarLineSegment(*tLineSegment));
            continue;
        }

        tArcSegment = dynamic_cast<const KGPlanarArcSegment*>(tElement);
        if (tArcSegment != nullptr) {
            fElements.push_back(new KGPlanarArcSegment(*tArcSegment));
            continue;
        }
    }

    return;
}

void KGPlanarPolyLine::StartPoint(const KTwoVector& aPoint)
{
    shapemsg_debug("adding first point to a planar poly line" << eom);
    fInitialized = false;

    const KGPlanarOpenPath* tElement;
    for (auto& element : fElements) {
        tElement = element;
        delete tElement;
    }
    fElements.clear();
    fStart = aPoint;
    fEnd = aPoint;

    return;
}
void KGPlanarPolyLine::NextLine(const KTwoVector& aVertex, const unsigned int aCount, const double aPower)
{
    shapemsg_debug("adding next line to a planar poly line" << eom);
    fInitialized = false;

    fElements.push_back(new KGPlanarLineSegment(fEnd, aVertex, aCount, aPower));
    fEnd = aVertex;

    return;
}
void KGPlanarPolyLine::NextArc(const KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong,
                               const unsigned int aCount)
{
    shapemsg_debug("adding next arc to a planar poly line" << eom);
    fInitialized = false;

    fElements.push_back(new KGPlanarArcSegment(fEnd, aVertex, aRadius, aLeft, aLong, aCount));
    fEnd = aVertex;

    return;
}
void KGPlanarPolyLine::PreviousLine(const KTwoVector& aVertex, const unsigned int aCount, const double aPower)
{
    shapemsg_debug("adding previous line to a planar poly line" << eom);
    fInitialized = false;

    fElements.push_back(new KGPlanarLineSegment(aVertex, fStart, aCount, aPower));
    fStart = aVertex;

    return;
}
void KGPlanarPolyLine::PreviousArc(const KTwoVector& aVertex, const double& aRadius, const bool& aLeft,
                                   const bool& aLong, const unsigned int aCount)
{
    shapemsg_debug("adding previous arc to a planar poly line" << eom);
    fInitialized = false;

    fElements.push_back(new KGPlanarArcSegment(aVertex, fStart, aRadius, aLeft, aLong, aCount));
    fStart = aVertex;

    return;
}

const KGPlanarPolyLine::Set& KGPlanarPolyLine::Elements() const
{
    if (fInitialized == false) {
        Initialize();
    }

    return fElements;
}

const double& KGPlanarPolyLine::Length() const
{
    if (fInitialized == false) {
        Initialize();
    }

    return fLength;
}
const KTwoVector& KGPlanarPolyLine::Centroid() const
{
    if (fInitialized == false) {
        Initialize();
    }

    return fCentroid;
}
const KTwoVector& KGPlanarPolyLine::Start() const
{
    if (fInitialized == false) {
        Initialize();
    }

    return fStart;
}
const KTwoVector& KGPlanarPolyLine::End() const
{
    if (fInitialized == false) {
        Initialize();
    }

    return fEnd;
}

KTwoVector KGPlanarPolyLine::At(const double& aLength) const
{
    if (fInitialized == false) {
        Initialize();
    }

    double tLength = aLength;

    if (tLength < 0.) {
        return fStart;
    }
    if (tLength > fLength) {
        return fEnd;
    }

    for (const auto* element : fElements) {
        if (element->Length() > tLength) {
            return element->At(tLength);
        }
        tLength -= element->Length();
    }
    return fEnd;
}

KTwoVector KGPlanarPolyLine::Point(const KTwoVector& aQuery) const
{
    if (fInitialized == false) {
        Initialize();
    }

    KTwoVector tCurrentPoint;
    double tCurrentDistance;

    KTwoVector tNearestPoint;
    double tNearestDistance;

    auto tIt = fElements.begin();

    tNearestPoint = (*tIt)->Point(aQuery);
    tNearestDistance = (tNearestPoint - aQuery).Magnitude();
    tIt++;

    while (tIt != fElements.end()) {
        tCurrentPoint = (*tIt)->Point(aQuery);
        tCurrentDistance = (tCurrentPoint - aQuery).Magnitude();

        if (tCurrentDistance < tNearestDistance) {
            tNearestPoint = tCurrentPoint;
            tNearestDistance = tCurrentDistance;
        }

        tIt++;
    }

    return tNearestPoint;
}
KTwoVector KGPlanarPolyLine::Normal(const KTwoVector& aQuery) const
{
    if (fInitialized == false) {
        Initialize();
    }

    KTwoVector tFirstPoint;
    KTwoVector tFirstNormal;
    double tFirstDistance;

    KTwoVector tSecondPoint;
    KTwoVector tSecondNormal;
    double tSecondDistance;

    KTwoVector tAveragePoint;
    KTwoVector tAverageNormal;
    double tAverageDistance;

    KTwoVector tNearestPoint;
    KTwoVector tNearestNormal;
    double tNearestDistance;

    auto tIt = fElements.begin();

    tFirstPoint = (*tIt)->Point(aQuery);
    tFirstNormal = (*tIt)->Normal(aQuery);
    tFirstDistance = (aQuery - tFirstPoint).Magnitude();

    tNearestPoint = tFirstPoint;
    tNearestNormal = tFirstNormal;
    tNearestDistance = tFirstDistance;

    tIt++;

    for (; tIt != fElements.end(); tIt++) {
        tSecondPoint = (*tIt)->Point(aQuery);
        tSecondNormal = (*tIt)->Normal(aQuery);
        tSecondDistance = (aQuery - tSecondPoint).Magnitude();

        tAveragePoint = .5 * (tFirstPoint + tSecondPoint);
        tAverageNormal = (tFirstNormal + tSecondNormal).Unit();
        tAverageDistance = .5 * (tFirstDistance + tSecondDistance);

        if (((tFirstPoint - tSecondPoint).Magnitude() / (tAveragePoint).Magnitude()) < 1.e-12) {
            if ((fabs(tAverageDistance - tNearestDistance) / tNearestDistance) < 1.e-12) {
                tNearestPoint = tAveragePoint;
                if (tAverageNormal.Dot(aQuery - tAveragePoint) > 0.) {
                    tNearestNormal = 1. * (aQuery - tAveragePoint).Unit();
                }
                else {
                    tNearestNormal = -1. * (aQuery - tAveragePoint).Unit();
                }
                tNearestDistance = tAverageDistance;

                tFirstPoint = tSecondPoint;
                tFirstNormal = tSecondNormal;
                tFirstDistance = tSecondDistance;
                continue;
            }

            if (tAverageDistance < tNearestDistance) {
                tNearestPoint = tAveragePoint;
                if (tAverageNormal.Dot(aQuery - tAveragePoint) > 0.) {
                    tNearestNormal = 1. * (aQuery - tAveragePoint).Unit();
                }
                else {
                    tNearestNormal = -1. * (aQuery - tAveragePoint).Unit();
                }
                tNearestDistance = tAverageDistance;

                tFirstPoint = tSecondPoint;
                tFirstNormal = tSecondNormal;
                tFirstDistance = tSecondDistance;
                continue;
            }
        }

        if (tSecondDistance < tNearestDistance) {
            tNearestPoint = tSecondPoint;
            tNearestNormal = tSecondNormal;
            tNearestDistance = tSecondDistance;

            tFirstPoint = tSecondPoint;
            tFirstNormal = tSecondNormal;
            tFirstDistance = tSecondDistance;
            continue;
        }

        tFirstPoint = tSecondPoint;
        tFirstNormal = tSecondNormal;
        tFirstDistance = tSecondDistance;
    }

    return tNearestNormal;
}
bool KGPlanarPolyLine::Above(const KTwoVector& aQuery) const
{
    if (fInitialized == false) {
        Initialize();
    }

    KTwoVector tPoint = Point(aQuery);
    KTwoVector tNormal = Normal(aQuery);

    if (tNormal.Dot(aQuery - tPoint) > 0.) {
        return true;
    }

    return false;
}

void KGPlanarPolyLine::Initialize() const
{
    shapemsg_debug("initializing a planar poly line" << eom);

    fLength = 0.;
    fCentroid.X() = 0;
    fCentroid.Y() = 0;

    for (const auto* element : fElements) {
        fLength += element->Length();
        fCentroid += element->Length() * element->Centroid();
    }
    fCentroid /= fLength;

    fInitialized = true;

    return;
}

}  // namespace KGeoBag
