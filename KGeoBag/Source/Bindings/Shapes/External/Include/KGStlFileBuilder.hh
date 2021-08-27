/**
 * @file KGStlFileBuilder.hh
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2021-07-02
 */

#ifndef KGSTLFILEBUILDER_HH_
#define KGSTLFILEBUILDER_HH_

#include "KComplexElement.hh"
#include "KContainer.hh"
#include "KGWrappedSurface.hh"
#include "KGWrappedSpace.hh"
#include "KGStlFile.hh"

using namespace KGeoBag;

namespace katrin
{

using KGStlFileBuilder = KComplexElement<KGStlFile>;

template<> inline bool KGStlFileBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "file") {
        anAttribute->CopyTo(fObject, &KGStlFile::SetFile);
        return true;
    }
    if (anAttribute->GetName() == "path") {
        anAttribute->CopyTo(fObject, &KGStlFile::SetPath);
        return true;
    }
    if (anAttribute->GetName() == "mesh_count") {
        anAttribute->CopyTo(fObject, &KGStlFile::SetNDisc);
        return true;
    }
    if (anAttribute->GetName() == "scale") {
        anAttribute->CopyTo(fObject, &KGStlFile::SetScaleFactor);
        return true;
    }
    if (anAttribute->GetName() == "selector") {
        // allowed syntax pattern: "a-b;c-d;..."
        for (auto & sel : KStringUtils::SplitBySingleDelim(anAttribute->AsString(), ";")) {
            size_t pos = sel.find_first_of("-");
            size_t first = 0, last = 0;
            if (pos == std::string::npos) {
                if (KStringUtils::Convert(sel, first))
                    fObject->SelectCell(first);
            }
            else {
                if (! KStringUtils::Convert(sel.substr(0, pos), first))
                    first = 0;
                if (! KStringUtils::Convert(sel.substr(pos+1), last))
                    last = std::numeric_limits<size_t>::max();
                fObject->SelectCellRange(first, last);
            }
        }
        return true;
    }
    return false;
}

using KGStlFileSurfaceBuilder = KComplexElement<KGWrappedSurface<KGStlFile>>;

template<> inline bool KGStlFileSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSurface<KGStlFile>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGStlFileSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "stl_file") {
        KGStlFile* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGStlFile> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}


using KGStlFileSpaceBuilder = KComplexElement<KGWrappedSpace<KGStlFile>>;

template<> inline bool KGStlFileSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSpace<KGStlFile>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGStlFileSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "stl_file") {
        KGStlFile* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGStlFile> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif  // KGSTLFILEBUILDER_HH_
