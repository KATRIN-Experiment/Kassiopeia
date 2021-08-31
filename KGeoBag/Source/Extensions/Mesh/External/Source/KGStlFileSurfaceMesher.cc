/**
 * @file KGStlFileSurfaceMesher.cc
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2021-07-02
 */

#include "KGStlFileSurfaceMesher.hh"

using namespace KGeoBag;

void KGStlFileSurfaceMesher::VisitWrappedSurface(KGStlFileSurface* stlSurface)
{
    auto object = stlSurface->GetObject();

    auto nElements = object->GetNumElements();
    coremsg(eInfo) << "Adding <" << nElements << "> surface elements to the mesh" << eom;

    fCurrentElements->reserve(fCurrentElements->size() + nElements);

    for (auto & elem : object->GetElements()) {
        auto t = new KGMeshTriangle(elem);

        if (object->GetNDisc() < 2)
            AddElement(t, false);
        else
            RefineAndAddElement(t, object->GetNDisc(), 1);
    }
}
