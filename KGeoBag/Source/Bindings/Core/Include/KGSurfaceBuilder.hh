#ifndef KGSURFACEBUILDER_HH_
#define KGSURFACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGCore.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGSurface> KGSurfaceBuilder;

template<> inline bool KGSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    using namespace KGeoBag;

    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "node") {
        KGSurface* tSource = KGInterface::GetInstance()->RetrieveSurface(anAttribute->AsReference<std::string>());
        if (tSource == nullptr) {
            return false;
        }

        KGSurface* tClone = tSource->CloneNode();
        tClone->SetName(fObject->GetName());
        tClone->AddTags(fObject->GetTags());

        fObject = tClone;
        Set(fObject);

        return true;
    }
    return false;
}

template<> inline bool KGSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "transformation") {
        fObject->Transform(anElement->AsPointer<KGeoBag::KTransformation>());
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
