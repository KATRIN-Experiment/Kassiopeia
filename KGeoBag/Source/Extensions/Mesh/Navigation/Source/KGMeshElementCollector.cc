#include "KGMeshElementCollector.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"

namespace KGeoBag
{


KGMeshElementCollector::KGMeshElementCollector() :
    fMeshContainer(nullptr),
    fOrigin(KThreeVector::sZero),
    fXAxis(KThreeVector::sXUnit),
    fYAxis(KThreeVector::sYUnit),
    fZAxis(KThreeVector::sZUnit),
    fCurrentOrigin(KThreeVector::sZero),
    fCurrentXAxis(KThreeVector::sXUnit),
    fCurrentYAxis(KThreeVector::sYUnit),
    fCurrentZAxis(KThreeVector::sZUnit)
{}

KGMeshElementCollector::~KGMeshElementCollector(){};

void KGMeshElementCollector::SetSystem(const KThreeVector& anOrigin, const KThreeVector& aXAxis,
                                       const KThreeVector& aYAxis, const KThreeVector& aZAxis)
{
    fOrigin = anOrigin;
    fXAxis = aXAxis;
    fYAxis = aYAxis;
    fZAxis = aZAxis;
    return;
}
const KThreeVector& KGMeshElementCollector::GetOrigin() const
{
    return fOrigin;
}
const KThreeVector& KGMeshElementCollector::GetXAxis() const
{
    return fXAxis;
}
const KThreeVector& KGMeshElementCollector::GetYAxis() const
{
    return fYAxis;
}
const KThreeVector& KGMeshElementCollector::GetZAxis() const
{
    return fZAxis;
}

void KGMeshElementCollector::VisitSurface(KGSurface* aSurface)
{
    fCurrentOrigin = aSurface->GetOrigin();
    fCurrentXAxis = aSurface->GetXAxis();
    fCurrentYAxis = aSurface->GetYAxis();
    fCurrentZAxis = aSurface->GetZAxis();
    fCurrentSurface = aSurface;
    fCurrentSpace = nullptr;
    Add(aSurface->AsExtension<KGMesh>());
}

void KGMeshElementCollector::VisitSpace(KGSpace* aSpace)
{
    fCurrentOrigin = aSpace->GetOrigin();
    fCurrentXAxis = aSpace->GetXAxis();
    fCurrentYAxis = aSpace->GetYAxis();
    fCurrentZAxis = aSpace->GetZAxis();
    fCurrentSpace = aSpace;
    fCurrentSurface = nullptr;
    Add(aSpace->AsExtension<KGMesh>());
}


KThreeVector KGMeshElementCollector::LocalToInternal(const KThreeVector& aVector)
{
    KThreeVector tGlobalVector(fCurrentOrigin + aVector.X() * fCurrentXAxis + aVector.Y() * fCurrentYAxis +
                               aVector.Z() * fCurrentZAxis);
    KThreeVector tInternalVector((tGlobalVector - fOrigin).Dot(fXAxis),
                                 (tGlobalVector - fOrigin).Dot(fYAxis),
                                 (tGlobalVector - fOrigin).Dot(fZAxis));
    return KThreeVector(tInternalVector.X(), tInternalVector.Y(), tInternalVector.Z());
}

void KGMeshElementCollector::Add(KGMeshData* aData)
{
    KGMeshElement* tMeshElement;
    KGMeshTriangle* tMeshTriangle;
    KGMeshRectangle* tMeshRectangle;
    KGMeshWire* tMeshWire;

    PreCollectionAction(aData);

    if (aData != nullptr) {
        if (aData->HasData()) {
            for (auto tElementIt = aData->Elements()->begin(); tElementIt != aData->Elements()->end(); tElementIt++) {
                tMeshElement = *tElementIt;

                tMeshTriangle = dynamic_cast<KGMeshTriangle*>(tMeshElement);
                if ((tMeshTriangle != nullptr)) {
                    fCurrentElementType = eTriangle;
                    //transform mesh triangle into global coordinates
                    KGMeshTriangle* t = new KGMeshTriangle(LocalToInternal(tMeshTriangle->GetP0()),
                                                           LocalToInternal(tMeshTriangle->GetP1()),
                                                           LocalToInternal(tMeshTriangle->GetP2()));
                    auto* navi_mesh_element = new KGNavigableMeshElement();
                    navi_mesh_element->SetMeshElement(t);
                    // navi_mesh_element->SetParentSurface(fCurrentSurface);
                    // navi_mesh_element->SetParentSpace(fCurrentSpace);
                    fMeshContainer->Add(navi_mesh_element);
                    PostCollectionAction(navi_mesh_element);
                    continue;
                }

                tMeshRectangle = dynamic_cast<KGMeshRectangle*>(tMeshElement);
                if ((tMeshRectangle != nullptr)) {
                    fCurrentElementType = eRectangle;
                    //transform mesh rectangle into global coordinates
                    KGMeshRectangle* r = new KGMeshRectangle(LocalToInternal(tMeshRectangle->GetP0()),
                                                             LocalToInternal(tMeshRectangle->GetP1()),
                                                             LocalToInternal(tMeshRectangle->GetP2()),
                                                             LocalToInternal(tMeshRectangle->GetP3()));
                    auto* navi_mesh_element = new KGNavigableMeshElement();
                    navi_mesh_element->SetMeshElement(r);
                    // navi_mesh_element->SetParentSurface(fCurrentSurface);
                    // navi_mesh_element->SetParentSpace(fCurrentSpace);
                    fMeshContainer->Add(navi_mesh_element);
                    PostCollectionAction(navi_mesh_element);
                    continue;
                }

                tMeshWire = dynamic_cast<KGMeshWire*>(tMeshElement);
                if ((tMeshWire != nullptr)) {
                    fCurrentElementType = eWire;
                    //transform mesh wire into global coordinates
                    KGMeshWire* w = new KGMeshWire(LocalToInternal(tMeshWire->GetP0()),
                                                   LocalToInternal(tMeshWire->GetP1()),
                                                   tMeshWire->GetDiameter());
                    auto* navi_mesh_element = new KGNavigableMeshElement();
                    navi_mesh_element->SetMeshElement(w);
                    // navi_mesh_element->SetParentSurface(fCurrentSurface);
                    // navi_mesh_element->SetParentSpace(fCurrentSpace);
                    fMeshContainer->Add(navi_mesh_element);
                    PostCollectionAction(navi_mesh_element);
                    continue;
                }
            }
        }
        // else
        // {
        //     warning about the surface not containing mesh data (all surfaces/spaces must be meshed)
        //     before being visited by the mesh element collector
        // }
    }

    return;
}


void KGMeshElementCollector::PreCollectionActionExecute(KGMeshData* /*aData*/){};

void KGMeshElementCollector::PostCollectionActionExecute(KGNavigableMeshElement* /*element */){};


}  // namespace KGeoBag
