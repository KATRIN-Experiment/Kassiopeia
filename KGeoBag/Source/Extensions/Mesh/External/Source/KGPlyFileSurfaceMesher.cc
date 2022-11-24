/**
 * @file KGPlyFileSurfaceMesher.cc
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2022-11-24
 */

#include "KGPlyFileSurfaceMesher.hh"

using namespace KGeoBag;

void KGPlyFileSurfaceMesher::VisitWrappedSurface(KGPlyFileSurface* PlySurface)
{
    auto object = PlySurface->GetObject();

    auto nElements = object->GetNumElements();
    coremsg(eInfo) << "Adding <" << nElements << "> surface elements to the mesh" << eom;

    fCurrentElements->reserve(fCurrentElements->size() + nElements);

    for (auto & elem : object->GetTriangles()) {
        auto t = new KGMeshTriangle(elem);

        if (object->GetNDisc() < 2)
            AddElement(t, false);
        else
            RefineAndAddElement(t, object->GetNDisc(), 1);
    }

    for (auto & elem : object->GetRectangles()) {
        auto r = new KGMeshRectangle(elem);

        if (object->GetNDisc() < 2)
            AddElement(r, false);
        else
            RefineAndAddElement(r, object->GetNDisc(), 1, object->GetNDisc(), 1);
    }
}
