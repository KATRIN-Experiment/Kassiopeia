#include "KGCore.hh"
#include "KGCoreMessage.hh"

using namespace std;

using katrin::KTransformation;
using katrin::KThreeVector;

namespace KGeoBag
{

KGSurface::Visitor::Visitor() = default;
KGSurface::Visitor::~Visitor() = default;

KGSurface::KGSurface() :
    fParent(nullptr),
    fOrigin(KThreeVector::sZero),
    fXAxis(KThreeVector::sXUnit),
    fYAxis(KThreeVector::sYUnit),
    fZAxis(KThreeVector::sZUnit)
{}
KGSurface::KGSurface(KGArea* anArea) :
    fParent(nullptr),
    fOrigin(KThreeVector::sZero),
    fXAxis(KThreeVector::sXUnit),
    fYAxis(KThreeVector::sYUnit),
    fZAxis(KThreeVector::sZUnit),
    fArea(anArea)
{}
KGSurface::~KGSurface()
{
    //delete all extensions
    KGExtensibleSurface* tExtension;
    vector<KGExtensibleSurface*>::iterator tExtensionIt;
    for (tExtensionIt = fExtensions.begin(); tExtensionIt != fExtensions.end(); tExtensionIt++) {
        tExtension = *tExtensionIt;
        delete tExtension;
    }
}

//************
//structurable
//************

void KGSurface::Orphan()
{
    if (fParent != nullptr) {
        vector<KGSurface*>::iterator tIt;
        for (tIt = fParent->fBoundaries.begin(); tIt != fParent->fBoundaries.end(); tIt++) {
            if ((*tIt) == this) {
                fParent->fBoundaries.erase(tIt);
                fParent = nullptr;
                return;
            }
        }
        for (tIt = fParent->fChildSurfaces.begin(); tIt != fParent->fChildSurfaces.end(); tIt++) {
            if ((*tIt) == this) {
                fParent->fChildSurfaces.erase(tIt);
                fParent = nullptr;
                return;
            }
        }
    }
    return;
}

const KGSpace* KGSurface::GetParent() const
{
    return fParent;
}

std::string KGSurface::GetPath() const
{
    string tPath = GetName();
    const KGSpace* tParent = GetParent();
    if (tParent != nullptr && tParent != KGInterface::GetInstance()->Root()) {
        tPath = tParent->GetPath() + "/" + tPath;
    }
    return tPath;
}

//*************
//transformable
//*************

void KGSurface::Transform(const KTransformation* aTransform)
{
    //transform the local frame
    coremsg_debug("starting transformation on surface <" << GetName() << ">" << eom);
    coremsg_debug("transformation has rotation of " << aTransform->GetRotation() << ret);
    coremsg_debug("and displacement of " << aTransform->GetDisplacement() << eom);

    coremsg_debug("applying transformation on origin " << fOrigin << eom);
    aTransform->Apply(fOrigin);
    coremsg_debug("transformation on origin done, new value " << fOrigin << eom);

    coremsg_debug("applying rotation on x axis " << fXAxis << eom);
    aTransform->ApplyRotation(fXAxis);
    coremsg_debug("rotation on x axis done, new value " << fXAxis << eom);

    coremsg_debug("applying rotation on y axis " << fYAxis << eom);
    aTransform->ApplyRotation(fYAxis);
    coremsg_debug("rotation on y axis done, new value " << fYAxis << eom);

    coremsg_debug("applying rotation on z axis " << fZAxis << eom);
    aTransform->ApplyRotation(fZAxis);
    coremsg_debug("rotation on z axis done, new value " << fZAxis << eom);

#ifdef KGeoBag_ENABLE_DEBUG
    double tEulerAlpha, tEulerBeta, tEulerGamma;
    aTransform->GetRotation().GetEulerAnglesInDegrees(tEulerAlpha, tEulerBeta, tEulerGamma);
    coremsg_debug("new euler angles are " << tEulerAlpha << " " << tEulerBeta << " " << tEulerGamma << eom);
#endif

    return;
}

const KThreeVector& KGSurface::GetOrigin() const
{
    return fOrigin;
}
const KThreeVector& KGSurface::GetXAxis() const
{
    return fXAxis;
}
const KThreeVector& KGSurface::GetYAxis() const
{
    return fYAxis;
}
const KThreeVector& KGSurface::GetZAxis() const
{
    return fZAxis;
}

//********
//clonable
//********

KGSurface* KGSurface::CloneNode() const
{
    auto* tClone = new KGSurface();

    //copy name
    tClone->SetName(this->GetName());

    //copy tags
    tClone->fTags = fTags;

    //copy area
    tClone->fArea = fArea;

    //copy the frame
    tClone->fOrigin = fOrigin;
    tClone->fXAxis = fXAxis;
    tClone->fYAxis = fYAxis;
    tClone->fZAxis = fZAxis;

    //clone all extensions
    KGExtensibleSurface* tExtension;
    vector<KGExtensibleSurface*>::const_iterator tExtensionIt;
    for (tExtensionIt = fExtensions.begin(); tExtensionIt != fExtensions.end(); tExtensionIt++) {
        tExtension = *tExtensionIt;
        tClone->fExtensions.push_back(tExtension->Clone(tClone));
    }

    return tClone;
}

//*********
//visitable
//*********

void KGSurface::AcceptNode(KGVisitor* aVisitor)
{
    coremsg_debug("surface named <" << GetName() << "> is receiving a visitor" << eom)

        //try to visit surface
        auto* tSurfaceVisitor = dynamic_cast<KGSurface::Visitor*>(aVisitor);
    if (tSurfaceVisitor != nullptr) {
        coremsg_debug("surface named <" << GetName() << "> is accepting a visitor" << eom)
            tSurfaceVisitor->VisitSurface(this);
    }

    //visit all extensions
    KGExtensibleSurface* tExtension;
    vector<KGExtensibleSurface*>::iterator tIt;
    for (tIt = fExtensions.begin(); tIt != fExtensions.end(); tIt++) {
        tExtension = *tIt;
        tExtension->Accept(aVisitor);
    }

    //visit area
    if (fArea) {
        fArea->Accept(aVisitor);
    }

    return;
}

//*********
//navigable
//*********

void KGSurface::Area(const std::shared_ptr<KGArea>& anArea)
{
    fArea = anArea;
    return;
}
const std::shared_ptr<KGArea>& KGSurface::Area() const
{
    return fArea;
}

bool KGSurface::Above(const KThreeVector& aQueryPoint) const
{
    if (fArea) {
        KThreeVector tLocalQueryPoint;

        tLocalQueryPoint[0] = (aQueryPoint[0] - fOrigin[0]) * fXAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fXAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fXAxis[2];
        tLocalQueryPoint[1] = (aQueryPoint[0] - fOrigin[0]) * fYAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fYAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fYAxis[2];
        tLocalQueryPoint[2] = (aQueryPoint[0] - fOrigin[0]) * fZAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fZAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fZAxis[2];

        return fArea->Above(tLocalQueryPoint);
    }
    return true;
}
KThreeVector KGSurface::Point(const KThreeVector& aQueryPoint) const
{
    if (fArea) {
        KThreeVector tLocalQueryPoint;
        KThreeVector tLocalNearestPoint;
        KThreeVector tNearestPoint;

        tLocalQueryPoint[0] = (aQueryPoint[0] - fOrigin[0]) * fXAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fXAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fXAxis[2];
        tLocalQueryPoint[1] = (aQueryPoint[0] - fOrigin[0]) * fYAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fYAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fYAxis[2];
        tLocalQueryPoint[2] = (aQueryPoint[0] - fOrigin[0]) * fZAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fZAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fZAxis[2];

        tLocalNearestPoint = fArea->Point(tLocalQueryPoint);

        tNearestPoint[0] = fOrigin[0] + tLocalNearestPoint[0] * fXAxis[0] + tLocalNearestPoint[1] * fYAxis[0] +
                           tLocalNearestPoint[2] * fZAxis[0];
        tNearestPoint[1] = fOrigin[1] + tLocalNearestPoint[0] * fXAxis[1] + tLocalNearestPoint[1] * fYAxis[1] +
                           tLocalNearestPoint[2] * fZAxis[1];
        tNearestPoint[2] = fOrigin[2] + tLocalNearestPoint[0] * fXAxis[2] + tLocalNearestPoint[1] * fYAxis[2] +
                           tLocalNearestPoint[2] * fZAxis[2];

        return tNearestPoint;
    }
    return fOrigin;
}
KThreeVector KGSurface::Normal(const KThreeVector& aQueryPoint) const
{
    if (fArea) {
        KThreeVector tLocalQueryPoint;
        KThreeVector tLocalNearestNormal;
        KThreeVector tNearestNormal;

        tLocalQueryPoint[0] = (aQueryPoint[0] - fOrigin[0]) * fXAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fXAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fXAxis[2];
        tLocalQueryPoint[1] = (aQueryPoint[0] - fOrigin[0]) * fYAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fYAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fYAxis[2];
        tLocalQueryPoint[2] = (aQueryPoint[0] - fOrigin[0]) * fZAxis[0] + (aQueryPoint[1] - fOrigin[1]) * fZAxis[1] +
                              (aQueryPoint[2] - fOrigin[2]) * fZAxis[2];

        tLocalNearestNormal = fArea->Normal(tLocalQueryPoint);

        tNearestNormal[0] = tLocalNearestNormal[0] * fXAxis[0] + tLocalNearestNormal[1] * fYAxis[0] +
                            tLocalNearestNormal[2] * fZAxis[0];
        tNearestNormal[1] = tLocalNearestNormal[0] * fXAxis[1] + tLocalNearestNormal[1] * fYAxis[1] +
                            tLocalNearestNormal[2] * fZAxis[1];
        tNearestNormal[2] = tLocalNearestNormal[0] * fXAxis[2] + tLocalNearestNormal[1] * fYAxis[2] +
                            tLocalNearestNormal[2] * fZAxis[2];

        return tNearestNormal;
    }
    return fZAxis;
}

}  // namespace KGeoBag
