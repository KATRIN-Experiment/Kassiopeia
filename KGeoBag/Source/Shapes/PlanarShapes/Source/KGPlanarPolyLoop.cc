#include "KGPlanarPolyLoop.hh"

#include "KGShapeMessage.hh"

namespace KGeoBag
{

KGPlanarPolyLoop::KGPlanarPolyLoop() :
    fLength(0.),
    fCentroid(0., 0.),
    fAnchor(0., 0.),
    fInitialized(false),
    fIsCounterClockwise(false)
{}
KGPlanarPolyLoop::KGPlanarPolyLoop(const KGPlanarPolyLoop& aCopy) :
    fLength(aCopy.fLength),
    fCentroid(aCopy.fCentroid),
    fAnchor(aCopy.fAnchor),
    fInitialized(aCopy.fInitialized),
    fIsCounterClockwise(aCopy.fIsCounterClockwise)
{
    const KGPlanarOpenPath* tElement;
    const KGPlanarLineSegment* tLineSegment;
    const KGPlanarArcSegment* tArcSegment;
    for (const auto* element : aCopy.fElements) {
        tElement = element;

        tLineSegment = dynamic_cast<const KGPlanarLineSegment*>(tElement);
        if (tLineSegment != nullptr) {
            fElements.push_back(tLineSegment->Clone());
            continue;
        }

        tArcSegment = dynamic_cast<const KGPlanarArcSegment*>(tElement);
        if (tArcSegment != nullptr) {
            fElements.push_back(tArcSegment->Clone());
            continue;
        }
    }
}
KGPlanarPolyLoop::~KGPlanarPolyLoop()
{
    shapemsg_debug("destroying a planar poly loop" << eom);

    const KGPlanarOpenPath* tElement;
    for (auto& element : fElements) {
        tElement = element;
        delete tElement;
    }
}

KGPlanarPolyLoop* KGPlanarPolyLoop::Clone() const
{
    return new KGPlanarPolyLoop(*this);
}
void KGPlanarPolyLoop::CopyFrom(const KGPlanarPolyLoop& aCopy)
{
    fLength = aCopy.fLength;
    fCentroid = aCopy.fCentroid;
    fAnchor = aCopy.fAnchor;
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

void KGPlanarPolyLoop::StartPoint(const katrin::KTwoVector& aPoint)
{
    shapemsg_debug("adding first point to a planar poly line" << eom);
    fInitialized = false;

    const KGPlanarOpenPath* tElement;
    for (auto& element : fElements) {
        tElement = element;
        delete tElement;
    }
    fElements.clear();
    fAnchor = aPoint;

    return;
}
void KGPlanarPolyLoop::NextLine(const katrin::KTwoVector& aVertex, const unsigned int aCount, const double aPower)
{
    shapemsg_debug("adding next line to a planar poly line" << eom);
    fInitialized = false;

    if (fElements.empty() == true) {
        fElements.push_back(new KGPlanarLineSegment(fAnchor, aVertex, aCount, aPower));
    }
    else {
        fElements.push_back(new KGPlanarLineSegment(fElements.back()->End(), aVertex, aCount, aPower));
    }

    return;
}
void KGPlanarPolyLoop::NextArc(const katrin::KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong,
                               const unsigned int aCount)
{
    shapemsg_debug("adding next arc to a planar poly line" << eom);
    fInitialized = false;

    if (fElements.empty() == true) {
        fElements.push_back(new KGPlanarArcSegment(fAnchor, aVertex, aRadius, aLeft, aLong, aCount));
    }
    else {
        fElements.push_back(new KGPlanarArcSegment(fElements.back()->End(), aVertex, aRadius, aLeft, aLong, aCount));
    }

    return;
}
void KGPlanarPolyLoop::PreviousLine(const katrin::KTwoVector& aVertex, const unsigned int aCount, const double aPower)
{
    shapemsg_debug("adding previous line to a planar poly line" << eom);
    fInitialized = false;

    if (fElements.empty() == true) {
        fElements.push_back(new KGPlanarLineSegment(aVertex, fAnchor, aCount, aPower));
    }
    else {
        fElements.push_back(new KGPlanarLineSegment(aVertex, fElements.front()->Start(), aCount, aPower));
    }

    return;
}
void KGPlanarPolyLoop::PreviousArc(const katrin::KTwoVector& aVertex, const double& aRadius, const bool& aLeft,
                                   const bool& aLong, const unsigned int aCount)
{
    shapemsg_debug("adding previous arc to a planar poly line" << eom);
    fInitialized = false;

    if (fElements.empty() == true) {
        fElements.push_back(new KGPlanarArcSegment(aVertex, fAnchor, aRadius, aLeft, aLong, aCount));
    }
    else {
        fElements.push_back(new KGPlanarArcSegment(aVertex, fElements.front()->Start(), aRadius, aLeft, aLong, aCount));
    }

    return;
}
void KGPlanarPolyLoop::LastLine(const unsigned int aCount, const double aPower)
{
    shapemsg_debug("adding last line to a planar poly loop" << eom);
    fInitialized = false;
    fElements.push_back(new KGPlanarLineSegment(fElements.back()->End(), fElements.front()->Start(), aCount, aPower));
}
void KGPlanarPolyLoop::LastArc(const double& aRadius, const bool& aLeft, const bool& aLong, const unsigned int aCount)
{
    shapemsg_debug("adding last arc to a planar poly loop" << eom);
    fInitialized = false;
    fElements.push_back(
        new KGPlanarArcSegment(fElements.back()->End(), fElements.front()->Start(), aRadius, aLeft, aLong, aCount));
}

const KGPlanarPolyLoop::Set& KGPlanarPolyLoop::Elements() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fElements;
}

const double& KGPlanarPolyLoop::Length() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fLength;
}
const katrin::KTwoVector& KGPlanarPolyLoop::Centroid() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fCentroid;
}
const katrin::KTwoVector& KGPlanarPolyLoop::Anchor() const
{
    if (fInitialized == false) {
        Initialize();
    }
    return fAnchor;
}

katrin::KTwoVector KGPlanarPolyLoop::At(const double& aLength) const
{
    if (fInitialized == false) {
        Initialize();
    }

    double tLength = aLength;

    if (tLength < 0.) {
        return fAnchor;
    }
    if (tLength > fLength) {
        return fAnchor;
    }

    for (const auto* element : fElements) {
        if (element->Length() > tLength) {
            return element->At(tLength);
        }
        tLength -= element->Length();
    }
    return fAnchor;
}
katrin::KTwoVector KGPlanarPolyLoop::Point(const katrin::KTwoVector& aQuery) const
{
    if (fInitialized == false) {
        Initialize();
    }

    katrin::KTwoVector tCurrent;
    double tCurrentDistanceSquared;

    katrin::KTwoVector tNearest;
    double tNearestDistanceSquared;

    auto tIt = fElements.begin();

    tNearest = (*tIt)->Point(aQuery);
    tNearestDistanceSquared = (tNearest - aQuery).MagnitudeSquared();
    tIt++;

    while (tIt != fElements.end()) {
        tCurrent = (*tIt)->Point(aQuery);
        tCurrentDistanceSquared = (tCurrent - aQuery).MagnitudeSquared();
        if (tCurrentDistanceSquared < tNearestDistanceSquared) {
            tNearest = tCurrent;
            tNearestDistanceSquared = tCurrentDistanceSquared;
        }
        tIt++;
    }

    return tNearest;
}
katrin::KTwoVector KGPlanarPolyLoop::Normal(const katrin::KTwoVector& aQuery) const
{
    if (fInitialized == false) {
        Initialize();
    }

    katrin::KTwoVector tFirstPoint;
    katrin::KTwoVector tFirstNormal;
    double tFirstDistance;

    katrin::KTwoVector tSecondPoint;
    katrin::KTwoVector tSecondNormal;
    double tSecondDistance;

    katrin::KTwoVector tAveragePoint;
    katrin::KTwoVector tAverageNormal;
    double tAverageDistance;

    katrin::KTwoVector tNearestPoint;
    katrin::KTwoVector tNearestNormal;
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

    if (fIsCounterClockwise) {
        return tNearestNormal;
    }
    else {
        return -1 * tNearestNormal;
    }
}
bool KGPlanarPolyLoop::Above(const katrin::KTwoVector& aQuery) const
{
    if (fInitialized == false) {
        Initialize();
    }

    katrin::KTwoVector tPoint = Point(aQuery);
    katrin::KTwoVector tNormal = Normal(aQuery);

    if (tNormal.Dot(aQuery - tPoint) > 0.) {
        return true;
    }

    return false;
}

void KGPlanarPolyLoop::Initialize() const
{
    shapemsg_debug("initializing a planar poly loop" << eom);

    fAnchor = fElements.front()->Start();

    fLength = 0.;
    fCentroid.X() = 0;
    fCentroid.Y() = 0;

    for (const auto* element : fElements) {
        fLength += element->Length();
        fCentroid += element->Length() * element->Centroid();
    }
    fCentroid /= fLength;

    fIsCounterClockwise = DetermineInteriorSide();

    fInitialized = true;

    return;
}

bool KGPlanarPolyLoop::DetermineInteriorSide() const
{
    katrin::KTwoVector first;
    katrin::KTwoVector second;
    double costheta;
    double theta;
    double total_angle = 0;

    std::vector<CIt> iters;
    CIt tItpp;
    for (auto tIt = fElements.begin(); tIt != fElements.end(); tIt++) {
        tItpp = tIt;
        tItpp++;
        if (tItpp != fElements.end()) {
            //this should work for arcs too as long
            //as the ploy loop is not self-intersecting!
            first = (*tIt)->End() - (*tIt)->Start();
            second = (*tItpp)->End() - (*tItpp)->Start();

            costheta = first * second;
            theta = std::fabs(std::acos(costheta));
            if ((first ^ second) > 0) {
                //we have a left hand turn
                total_angle += theta;
            }
            else {
                //we have a right hand turn
                total_angle -= theta;
            }
        }
    }

    //convention is as view from above xy plane, looking in -z direction
    if (total_angle > 0) {
        return true;  //loop runs counter-clockwise
    }
    else {
        return false;  //loop runs clockwise
    }
}

}  // namespace KGeoBag
