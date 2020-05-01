#include "KGNavigableMeshElement.hh"

namespace KGeoBag
{


KGNavigableMeshElement::KGNavigableMeshElement()
{
    fType = -1;
    fMeshElement = nullptr;
    fID = -1;
}

KGNavigableMeshElement::~KGNavigableMeshElement()
{
    delete fMeshElement;
};

void KGNavigableMeshElement::SetMeshElement(KGMeshTriangle* triangle)
{
    fMeshElement = triangle;
    fType = KGMESH_TRIANGLE_ID;
}

void KGNavigableMeshElement::SetMeshElement(KGMeshRectangle* rectange)
{
    fMeshElement = rectange;
    fType = KGMESH_RECTANGLE_ID;
}

void KGNavigableMeshElement::SetMeshElement(KGMeshWire* wire)
{
    fMeshElement = wire;
    fType = KGMESH_WIRE_ID;
}

short KGNavigableMeshElement::GetMeshElementType()
{
    return fType;
}

const KGMeshElement* KGNavigableMeshElement::GetMeshElement() const
{
    return fMeshElement;
}

}  // namespace KGeoBag
