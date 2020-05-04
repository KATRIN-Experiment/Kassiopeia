#include "KGNavigableMeshElementContainer.hh"

namespace KGeoBag
{

KGNavigableMeshElementContainer::KGNavigableMeshElementContainer()
{
    fValidGlobalCube = false;
    fMeshElements.clear();
}

KGNavigableMeshElementContainer::~KGNavigableMeshElementContainer()
{
    for (unsigned int i = 0; i < fMeshElements.size(); i++) {
        delete fMeshElements[i];
    }
    fMeshElements.clear();
}


//access individual element data
void KGNavigableMeshElementContainer::Add(KGNavigableMeshElement* element)
{
    element->SetID(fMeshElements.size());  //set unique id
    fMeshElements.push_back(element);      //add to the collection

    //compute the bounding ball
    KGPointCloud<KGMESH_DIM> point_cloud = element->GetMeshElement()->GetPointCloud();
    fBoundaryCalculator.Reset();
    fBoundaryCalculator.AddPointCloud(&point_cloud);
    fMeshElementBoundingBalls.push_back(fBoundaryCalculator.GetMinimalBoundingBall());

    fValidGlobalCube = false;
}

KGNavigableMeshElement* KGNavigableMeshElementContainer::GetElement(unsigned int id)
{
    return fMeshElements[id];
}

KGBall<KGMESH_DIM> KGNavigableMeshElementContainer::GetElementBoundingBall(unsigned int id)
{
    return fMeshElementBoundingBalls[id];
}

KGCube<KGMESH_DIM> KGNavigableMeshElementContainer::GetGlobalBoundingCube()
{

    if (!fValidGlobalCube) {
        fBoundaryCalculator.Reset();
        for (unsigned int i = 0; i < fMeshElements.size(); i++) {
            //KGPointCloud<KGMESH_DIM> point_cloud = fMeshElements[i]->GetMeshElement()->GetPointCloud();
            //fBoundaryCalculator.AddPointCloud(&point_cloud);
            fBoundaryCalculator.AddBall(&(fMeshElementBoundingBalls[i]));
        }
        fGlobalBoundingCube = fBoundaryCalculator.GetMinimalBoundingCube();
        fValidGlobalCube = true;
    }

    return fGlobalBoundingCube;
}


}  // namespace KGeoBag
