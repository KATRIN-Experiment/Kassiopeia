/**
 * @file KGPlyFileBuilder.hh
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2022-11-24
 */

#ifndef KGPLYFILEBUILDER_HH_
#define KGPLYFILEBUILDER_HH_

#include "KComplexElement.hh"
#include "KContainer.hh"
#include "KGWrappedSurface.hh"
#include "KGWrappedSpace.hh"
#include "KGPlyFile.hh"

#include "KBaseStringUtils.h"
#include "KException.h"

using namespace KGeoBag;

namespace katrin
{

using KGPlyFileBuilder = KComplexElement<KGPlyFile>;

template<> inline bool KGPlyFileBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "file") {
        anAttribute->CopyTo(fObject, &KGPlyFile::SetFile);
        return true;
    }
    if (anAttribute->GetName() == "path") {
        anAttribute->CopyTo(fObject, &KGPlyFile::SetPath);
        return true;
    }
    if (anAttribute->GetName() == "mesh_count") {
        anAttribute->CopyTo(fObject, &KGPlyFile::SetNDisc);
        return true;
    }
    if (anAttribute->GetName() == "scale") {
        anAttribute->CopyTo(fObject, &KGPlyFile::SetScaleFactor);
        return true;
    }
    if (anAttribute->GetName() == "selector") {
        // allowed syntax pattern: "a-b;c-d;..."
        for (std::string& sel : KBaseStringUtils::SplitTrimAndConvert<std::string>(anAttribute->AsString(), ";, ")) {
            size_t pos = sel.find_first_of("-");
            size_t first = 0, last = -1;
            if (pos == std::string::npos) {
                first = KBaseStringUtils::Convert<size_t>(sel);
                fObject->SelectCell(first);
            }
            else {
                if (pos > 0)
                    first = KBaseStringUtils::Convert<size_t>(sel.substr(0, pos));
                if (pos+1 < sel.length())
                    last = KBaseStringUtils::Convert<size_t>(sel.substr(pos+1));
                fObject->SelectCellRange(first, last);
            }
        }
        return true;
    }
    return false;
}

using KGPlyFileSurfaceBuilder = KComplexElement<KGWrappedSurface<KGPlyFile>>;

template<> inline bool KGPlyFileSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSurface<KGPlyFile>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGPlyFileSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "ply_file") {
        KGPlyFile* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGPlyFile> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}


using KGPlyFileSpaceBuilder = KComplexElement<KGWrappedSpace<KGPlyFile>>;

template<> inline bool KGPlyFileSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSpace<KGPlyFile>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGPlyFileSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "ply_file") {
        KGPlyFile* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGPlyFile> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif  // KGPlyFILEBUILDER_HH_
