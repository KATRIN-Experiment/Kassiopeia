/**
 * @file KGStlFileSpaceMesher.cc
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2021-07-02
 */

#include "KGStlFileSpaceMesher.hh"

using namespace KGeoBag;

void KGStlFileSpaceMesher::VisitWrappedSpace(KGStlFileSpace* stlSpace)
{
    auto object = stlSpace->GetObject();

    auto nElements = object->GetNumSolidElements();
    coremsg(eInfo) << "Adding <" << nElements << "> solid surface elements to the mesh" << eom;

    fCurrentElements->reserve(fCurrentElements->size() + nElements);

    for (auto & solid : object->GetSolids()) {
        for (auto & elem : solid) {
            auto t = new KGMeshTriangle(elem);

            if (object->GetNDisc() < 2)
                AddElement(t, false);
            else
                RefineAndAddElement(t, object->GetNDisc(), 1);
        }
    }
}
