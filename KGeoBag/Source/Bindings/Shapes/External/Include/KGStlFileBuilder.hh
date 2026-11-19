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

#include "KBaseStringUtils.h"
#include "KException.h"

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
        for (std::string& sel : KBaseStringUtils::SplitTrimAndConvert<std::string>(anAttribute->AsString(), ";, ")) {
            size_t pos = sel.find_first_of("-");
            size_t first = 0, last = 0;
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

//! @image Documentation/gh-pages/source/_images/kgeobag_teapot_mesh.png 400pt
//!
//! - Works with GeometryViewer: Yes.
//! - Works with MeshViewer: Yes.
//!
//! An XML example is as follows:
//!
//! .. code-block:: xml
//!
//!     <stl_file_surface name="cube_surface">
//!         <stl_file file="Menger_sponge.stl" scale="0.5" mesh_count="1"/>
//!     </stl_file_surface>
//!
//!     <stl_file_surface name="teapot_lid">
//!         <stl_file file="Utah_teapot.stl" selector="-2267"/>
//!     </stl_file_surface>
//!
//!     <stl_file_surface name="teapot_solid_body">
//!         <stl_file file="Utah_teapot.stl" selector="2268-"/>
//!     </stl_file_surface>
//!
//! Note that because the geometry is defined by an external file, the possible modifications in the *KGeoBag* framework
//! are limited. Besides the translations/rotations that can be applied to any object, one may also scale the external
//! geometry or increase its number of mesh elements. If the triangles in the file are sufficienctly ordered, then one
//! can use the ``selector`` attribute to specify indices of the triangles to be used in *KGeoBag*. In the case of the
//! ``Utah_teapot.stl`` example file, this allows to split the lid from the pot's body.
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


//! For visualization, see :ref:`<stl_file_surface> <binding-kgwrappedsurface-kgstlfile>`.
//!
//! - Works with GeometryViewer: Yes.
//! - Works with MeshViewer: Yes.
//! @example
//! <stl_file_space name="cube_surface">
//!     <stl_file file="Menger_sponge.stl" scale="0.5" mesh_count="1"/>
//! </stl_file_space>
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
