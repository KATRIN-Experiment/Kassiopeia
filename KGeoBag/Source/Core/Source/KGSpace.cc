#include "KGCore.hh"

#include <limits>

using namespace std;

namespace KGeoBag
{

KGSpace::Visitor::Visitor() {}
KGSpace::Visitor::~Visitor() {}

KGSpace::KGSpace() :
    fExtensions(),
    fParent(nullptr),
    fBoundaries(),
    fChildSurfaces(),
    fChildSpaces(),
    fOrigin(KThreeVector::sZero),
    fXAxis(KThreeVector::sXUnit),
    fYAxis(KThreeVector::sYUnit),
    fZAxis(KThreeVector::sZUnit),
    fVolume()
{}
KGSpace::KGSpace(KGVolume* aVolume) :
    fExtensions(),
    fParent(nullptr),
    fBoundaries(),
    fChildSurfaces(),
    fChildSpaces(),
    fOrigin(KThreeVector::sZero),
    fXAxis(KThreeVector::sXUnit),
    fYAxis(KThreeVector::sYUnit),
    fZAxis(KThreeVector::sZUnit),
    fVolume()
{
    Volume(std::shared_ptr<KGVolume>(aVolume));
}
KGSpace::~KGSpace()
{
    //delete all extensions
    KGExtensibleSpace* tExtension;
    vector<KGExtensibleSpace*>::iterator tIt;
    for (tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++) {
        tExtension = *tIt;
        delete tExtension;
    }

    //delete all boundaries
    KGSurface* tBoundary;
    vector<KGSurface*>::iterator tBoundaryIt;
    for (tBoundaryIt = fBoundaries.begin(); tBoundaryIt != fBoundaries.end(); tBoundaryIt++) {
        tBoundary = *tBoundaryIt;
        delete tBoundary;
    }

    //delete all child surfaces
    KGSurface* tSurface;
    vector<KGSurface*>::iterator tSurfaceIt;
    for (tSurfaceIt = fChildSurfaces.begin(); tSurfaceIt != fChildSurfaces.end(); tSurfaceIt++) {
        tSurface = *tSurfaceIt;
        delete tSurface;
    }

    //delete all child spaces
    KGSpace* tSpace;
    vector<KGSpace*>::iterator tSpaceIt;
    for (tSpaceIt = fChildSpaces.begin(); tSpaceIt != fChildSpaces.end(); tSpaceIt++) {
        tSpace = *tSpaceIt;
        delete tSpace;
    }
}

//************
//structurable
//************

void KGSpace::Orphan()
{
    if (fParent != nullptr) {
        vector<KGSpace*>::iterator tIt;
        for (tIt = fParent->fChildSpaces.begin(); tIt != fParent->fChildSpaces.end(); tIt++) {
            if ((*tIt) == this) {
                fParent->fChildSpaces.erase(tIt);
                fParent = nullptr;
                return;
            }
        }
    }
    return;
}

void KGSpace::AddBoundary(KGSurface* aBoundary)
{
    aBoundary->Orphan();
    aBoundary->fParent = this;
    this->fBoundaries.push_back(aBoundary);
    return;
}

void KGSpace::AddChildSurface(KGSurface* aSurface)
{
    aSurface->Orphan();
    aSurface->fParent = this;
    this->fChildSurfaces.push_back(aSurface);
    return;
}

void KGSpace::AddChildSpace(KGSpace* aSpace)
{
    aSpace->Orphan();
    aSpace->fParent = this;
    this->fChildSpaces.push_back(aSpace);
    return;
}

const KGSpace* KGSpace::GetParent() const
{
    return fParent;
}

std::string KGSpace::GetPath() const
{
    string tPath = GetName();
    const KGSpace* tParent = GetParent();
    while (tParent != nullptr && tParent != KGInterface::GetInstance()->Root()) {
        tPath = tParent->GetName() + "/" + tPath;
        tParent = tParent->GetParent();
    }
    return tPath;
}

const vector<KGSurface*>* KGSpace::GetBoundaries() const
{
    return &fBoundaries;
}

const vector<KGSurface*>* KGSpace::GetChildSurfaces() const
{
    return &fChildSurfaces;
}

const vector<KGSpace*>* KGSpace::GetChildSpaces() const
{
    return &fChildSpaces;
}

//*************
//transformable
//*************

void KGSpace::Transform(const KTransformation* aTransform)
{
    //transform the local frame
    coremsg_debug("starting transformation on space <" << GetName() << ">" << eom;)
        coremsg_debug("transformation has rotation of " << aTransform->GetRotation() << ret;)
            coremsg_debug("and displacement of " << aTransform->GetDisplacement() << eom;)

                coremsg_debug("applying transformation on origin " << fOrigin << eom;) aTransform->Apply(fOrigin);
    coremsg_debug("transformation on origin done, new value " << fOrigin << eom;)

        coremsg_debug("applying rotation on x axis " << fXAxis << eom;) aTransform->ApplyRotation(fXAxis);
    coremsg_debug("rotation on x axis done, new value " << fXAxis << eom;)

        coremsg_debug("applying rotation on y axis " << fYAxis << eom;) aTransform->ApplyRotation(fYAxis);
    coremsg_debug("rotation on y axis done, new value " << fYAxis << eom;)

        coremsg_debug("applying rotation on z axis " << fZAxis << eom;) aTransform->ApplyRotation(fZAxis);
    coremsg_debug("rotation on z axis done, new value " << fZAxis << eom;)

        //transform all the boundaries
        coremsg_debug("starting transformation on boundaries of <" << GetName() << ">" << eom;) KGSurface* tBoundary;
    vector<KGSurface*>::const_iterator tBoundaryIt;
    for (tBoundaryIt = fBoundaries.begin(); tBoundaryIt != fBoundaries.end(); tBoundaryIt++) {
        tBoundary = *tBoundaryIt;
        tBoundary->Transform(aTransform);
    }

    //transform all the child surfaces
    coremsg_debug("starting transformation on child surfaces of <" << GetName() << ">" << eom;) KGSurface* tSurface;
    vector<KGSurface*>::const_iterator tSurfaceIt;
    for (tSurfaceIt = fChildSurfaces.begin(); tSurfaceIt != fChildSurfaces.end(); tSurfaceIt++) {
        tSurface = *tSurfaceIt;
        tSurface->Transform(aTransform);
    }

    //transform all the child spaces
    coremsg_debug("starting transformation on child spaces of <" << GetName() << ">" << eom;) KGSpace* tSpace;
    vector<KGSpace*>::const_iterator tSpaceIt;
    for (tSpaceIt = fChildSpaces.begin(); tSpaceIt != fChildSpaces.end(); tSpaceIt++) {
        tSpace = *tSpaceIt;
        tSpace->Transform(aTransform);
    }

    return;
}

const KThreeVector& KGSpace::GetOrigin() const
{
    return fOrigin;
}
const KThreeVector& KGSpace::GetXAxis() const
{
    return fXAxis;
}
const KThreeVector& KGSpace::GetYAxis() const
{
    return fYAxis;
}
const KThreeVector& KGSpace::GetZAxis() const
{
    return fZAxis;
}

//********
//clonable
//********

KGSpace* KGSpace::CloneNode() const
{
    auto* tClone = new KGSpace();

    //copy name
    tClone->SetName(this->GetName());

    //copy tags
    tClone->fTags = fTags;

    //copy volume
    tClone->fVolume = fVolume;

    //copy the frame
    tClone->fOrigin = fOrigin;
    tClone->fXAxis = fXAxis;
    tClone->fYAxis = fYAxis;
    tClone->fZAxis = fZAxis;

    //clone all extensions
    KGExtensibleSpace* tExtension;
    vector<KGExtensibleSpace*>::const_iterator tExtensionIt;
    for (tExtensionIt = fExtensions.begin(); tExtensionIt != fExtensions.end(); tExtensionIt++) {
        tExtension = *tExtensionIt;
        tClone->fExtensions.push_back(tExtension->Clone(tClone));
    }

    //clone all boundaries
    KGSurface* tBoundary;
    vector<KGSurface*>::const_iterator tBoundaryIt;
    for (tBoundaryIt = fBoundaries.begin(); tBoundaryIt != fBoundaries.end(); tBoundaryIt++) {
        tBoundary = *tBoundaryIt;
        tClone->AddBoundary(tBoundary->CloneNode());
    }

    return tClone;
}

KGSpace* KGSpace::CloneTree() const
{
    auto* tClone = new KGSpace();

    //copy name
    tClone->SetName(this->GetName());

    //copy tags
    tClone->fTags = fTags;

    //copy volume
    tClone->fVolume = fVolume;

    //copy the frame
    tClone->fOrigin = fOrigin;
    tClone->fXAxis = fXAxis;
    tClone->fYAxis = fYAxis;
    tClone->fZAxis = fZAxis;

    //clone all extensions
    KGExtensibleSpace* tExtension;
    vector<KGExtensibleSpace*>::const_iterator tExtensionIt;
    for (tExtensionIt = fExtensions.begin(); tExtensionIt != fExtensions.end(); tExtensionIt++) {
        tExtension = *tExtensionIt;
        tClone->fExtensions.push_back(tExtension->Clone(tClone));
    }

    //clone all boundaries
    KGSurface* tBoundary;
    vector<KGSurface*>::const_iterator tBoundaryIt;
    for (tBoundaryIt = fBoundaries.begin(); tBoundaryIt != fBoundaries.end(); tBoundaryIt++) {
        tBoundary = *tBoundaryIt;
        tClone->AddBoundary(tBoundary->CloneNode());
    }

    //clone all child surfaces
    KGSurface* tSurface;
    vector<KGSurface*>::const_iterator tSurfaceIt;
    for (tSurfaceIt = fChildSurfaces.begin(); tSurfaceIt != fChildSurfaces.end(); tSurfaceIt++) {
        tSurface = *tSurfaceIt;
        tClone->AddChildSurface(tSurface->CloneNode());
    }

    //clone all child spaces
    KGSpace* tSpace;
    vector<KGSpace*>::const_iterator tSpaceIt;
    for (tSpaceIt = fChildSpaces.begin(); tSpaceIt != fChildSpaces.end(); tSpaceIt++) {
        tSpace = *tSpaceIt;
        tClone->AddChildSpace(tSpace->CloneTree());
    }

    return tClone;
}

//*********
//visitable
//*********

void KGSpace::AcceptNode(KGVisitor* aVisitor)
{
    coremsg_debug("space named <" << GetName() << "> is receiving a visitor" << eom)

        //try to visit the space
        auto* tSpaceVisitor = dynamic_cast<KGSpace::Visitor*>(aVisitor);
    if (tSpaceVisitor != nullptr) {
        coremsg_debug("space named <" << GetName() << "> is accepting a visitor" << eom)
            tSpaceVisitor->VisitSpace(this);
    }

    //visit all extensions
    KGExtensibleSpace* tExtension;
    vector<KGExtensibleSpace*>::iterator tIt;
    for (tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++) {
        tExtension = *tIt;

        tExtension->Accept(aVisitor);
    }

    //visit the volume
    if (fVolume) {
        fVolume->Accept(aVisitor);
    }

    return;
}

void KGSpace::AcceptTree(KGVisitor* aVisitor)
{
    coremsg_debug("space named <" << GetName() << "> is receiving a visitor" << eom)

        //try to visit the space
        auto* tSpaceVisitor = dynamic_cast<KGSpace::Visitor*>(aVisitor);
    if (tSpaceVisitor != nullptr) {
        coremsg_debug("space named <" << GetName() << "> is accepting a visitor" << eom)
            tSpaceVisitor->VisitSpace(this);
    }

    //visit all extensions
    KGExtensibleSpace* tExtension;
    vector<KGExtensibleSpace*>::iterator tIt;
    for (tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++) {
        tExtension = *tIt;
        tExtension->Accept(aVisitor);
    }

    //visit the volume
    if (fVolume) {
        fVolume->Accept(aVisitor);
    }

    //visit all boundaries
    KGSurface* tBoundary;
    vector<KGSurface*>::iterator tBoundaryIt;
    for (tBoundaryIt = fBoundaries.begin(); tBoundaryIt != fBoundaries.end(); tBoundaryIt++) {
        tBoundary = *tBoundaryIt;
        tBoundary->AcceptNode(aVisitor);
    }

    //visit all child surfaces
    KGSurface* tSurface;
    vector<KGSurface*>::iterator tSurfaceIt;
    for (tSurfaceIt = fChildSurfaces.begin(); tSurfaceIt != fChildSurfaces.end(); tSurfaceIt++) {
        tSurface = *tSurfaceIt;
        tSurface->AcceptNode(aVisitor);
    }

    //visit all child spaces
    KGSpace* tSpace;
    vector<KGSpace*>::iterator tSpaceIt;
    for (tSpaceIt = fChildSpaces.begin(); tSpaceIt != fChildSpaces.end(); tSpaceIt++) {
        tSpace = *tSpaceIt;
        tSpace->AcceptTree(aVisitor);
    }

    return;
}

//*********
//navigable
//*********

void KGSpace::Volume(const std::shared_ptr<KGVolume>& aVolume)
{
    //clear out old boundaries
    KGSurface* tBoundary;
    vector<KGSurface*>::iterator tBoundaryIt;

    for (tBoundaryIt = fBoundaries.begin(); tBoundaryIt != fBoundaries.end(); tBoundaryIt++) {
        tBoundary = *tBoundaryIt;
        delete tBoundary;
    }

    //put in new boundaries
    vector<std::shared_ptr<KGBoundary>>::const_iterator tAreaIt;
    for (tAreaIt = aVolume->Boundaries().begin(); tAreaIt != aVolume->Boundaries().end(); tAreaIt++) {
        tBoundary = new KGSurface();
        tBoundary->SetName((*tAreaIt)->GetName());
        tBoundary->SetTags((*tAreaIt)->GetTags());
        tBoundary->Area(std::static_pointer_cast<KGArea, KGBoundary>(*tAreaIt));  // FIXME this code looks ugly
        AddBoundary(tBoundary);
    }

    //replace volume
    fVolume = aVolume;

    return;
}
const std::shared_ptr<KGVolume>& KGSpace::Volume() const
{
    return fVolume;
}

bool KGSpace::Outside(const KThreeVector& aQueryPoint) const
{
    if (fVolume) {
        KThreeVector tLocalQueryPoint;

        tLocalQueryPoint[0] = (aQueryPoint[0] - fOrigin[0]) * fXAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fXAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fXAxis[2];
        tLocalQueryPoint[1] = (aQueryPoint[0] - fOrigin[0]) * fYAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fYAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fYAxis[2];
        tLocalQueryPoint[2] = (aQueryPoint[0] - fOrigin[0]) * fZAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fZAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fZAxis[2];

        return fVolume->Outside(tLocalQueryPoint);
    }
    else {
        KGSpace* tSpace;
        vector<KGSpace*>::const_iterator tSpaceIt;
        for (tSpaceIt = fChildSpaces.begin(); tSpaceIt != fChildSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            if (tSpace->Outside(aQueryPoint) == false) {
                return false;
            }
        }
        return true;
    }
}
KThreeVector KGSpace::Point(const KThreeVector& aQueryPoint) const
{
    if (fVolume) {
        KThreeVector tLocalQueryPoint;
        KThreeVector tLocalNearestPoint;
        KThreeVector tNearestPoint;

        tLocalQueryPoint[0] = (aQueryPoint[0] - fOrigin[0]) * fXAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fXAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fXAxis[2];
        tLocalQueryPoint[1] = (aQueryPoint[0] - fOrigin[0]) * fYAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fYAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fYAxis[2];
        tLocalQueryPoint[2] = (aQueryPoint[0] - fOrigin[0]) * fZAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fZAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fZAxis[2];

        tLocalNearestPoint = fVolume->Point(tLocalQueryPoint);

        tNearestPoint[0] = fOrigin[0] + tLocalNearestPoint[0] * fXAxis[0] + tLocalNearestPoint[1] * fYAxis[0] +
                           tLocalNearestPoint[2] * fZAxis[0];
        tNearestPoint[1] = fOrigin[1] + tLocalNearestPoint[0] * fXAxis[1] + tLocalNearestPoint[1] * fYAxis[1] +
                           tLocalNearestPoint[2] * fZAxis[1];
        tNearestPoint[2] = fOrigin[2] + tLocalNearestPoint[0] * fXAxis[2] + tLocalNearestPoint[1] * fYAxis[2] +
                           tLocalNearestPoint[2] * fZAxis[2];

        return tNearestPoint;
    }
    else {
        double tNearestDistanceSquared = std::numeric_limits<double>::max();
        KThreeVector tNearestPoint = fOrigin;

        double tCurrentDistanceSquared;
        KThreeVector tCurrentPoint;

        KGSurface* tSurface;
        vector<KGSurface*>::const_iterator tSurfaceIt;
        for (tSurfaceIt = fChildSurfaces.begin(); tSurfaceIt != fChildSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            tCurrentPoint = tSurface->Point(aQueryPoint);
            tCurrentDistanceSquared = (tCurrentPoint - aQueryPoint).MagnitudeSquared();
            if (tCurrentDistanceSquared < tNearestDistanceSquared) {
                tNearestDistanceSquared = tCurrentDistanceSquared;
                tNearestPoint = tCurrentPoint;
            }
        }

        KGSpace* tSpace;
        vector<KGSpace*>::const_iterator tSpaceIt;
        for (tSpaceIt = fChildSpaces.begin(); tSpaceIt != fChildSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            tCurrentPoint = tSpace->Point(aQueryPoint);
            tCurrentDistanceSquared = (tCurrentPoint - aQueryPoint).MagnitudeSquared();
            if (tCurrentDistanceSquared < tNearestDistanceSquared) {
                tNearestDistanceSquared = tCurrentDistanceSquared;
                tNearestPoint = tCurrentPoint;
            }
        }

        return tNearestPoint;
    }
}
KThreeVector KGSpace::Normal(const KThreeVector& aQueryPoint) const
{
    if (fVolume) {
        KThreeVector tLocalQueryPoint;
        KThreeVector tLocalNearestNormal;
        KThreeVector tNearestNormal;

        tLocalQueryPoint[0] = (aQueryPoint[0] - fOrigin[0]) * fXAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fXAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fXAxis[2];
        tLocalQueryPoint[1] = (aQueryPoint[0] - fOrigin[0]) * fYAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fYAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fYAxis[2];
        tLocalQueryPoint[2] = (aQueryPoint[0] - fOrigin[0]) * fZAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fZAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fZAxis[2];

        tLocalNearestNormal = fVolume->Normal(tLocalQueryPoint);

        tNearestNormal[0] = tLocalNearestNormal[0] * fXAxis[0] + tLocalNearestNormal[1] * fYAxis[0] +
                            tLocalNearestNormal[2] * fZAxis[0];
        tNearestNormal[1] = tLocalNearestNormal[0] * fXAxis[1] + tLocalNearestNormal[1] * fYAxis[1] +
                            tLocalNearestNormal[2] * fZAxis[1];
        tNearestNormal[2] = tLocalNearestNormal[0] * fXAxis[2] + tLocalNearestNormal[1] * fYAxis[2] +
                            tLocalNearestNormal[2] * fZAxis[2];

        return tNearestNormal;
    }
    else {
        double tNearestDistanceSquared = std::numeric_limits<double>::max();
        KThreeVector tNearestNormal = fZAxis;

        double tCurrentDistanceSquared;
        KThreeVector tCurrentPoint;

        KGSurface* tSurface;
        vector<KGSurface*>::const_iterator tSurfaceIt;
        for (tSurfaceIt = fChildSurfaces.begin(); tSurfaceIt != fChildSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            tCurrentPoint = tSurface->Point(aQueryPoint);
            tCurrentDistanceSquared = (tCurrentPoint - aQueryPoint).MagnitudeSquared();
            if (tCurrentDistanceSquared < tNearestDistanceSquared) {
                tNearestDistanceSquared = tCurrentDistanceSquared;
                tNearestNormal = tSurface->Normal(aQueryPoint);
            }
        }

        KGSpace* tSpace;
        vector<KGSpace*>::const_iterator tSpaceIt;
        for (tSpaceIt = fChildSpaces.begin(); tSpaceIt != fChildSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            tCurrentPoint = tSpace->Point(aQueryPoint);
            tCurrentDistanceSquared = (tCurrentPoint - aQueryPoint).MagnitudeSquared();
            if (tCurrentDistanceSquared < tNearestDistanceSquared) {
                tNearestDistanceSquared = tCurrentDistanceSquared;
                tNearestNormal = tSpace->Normal(aQueryPoint);
            }
        }

        return tNearestNormal;
    }
}

}  // namespace KGeoBag
