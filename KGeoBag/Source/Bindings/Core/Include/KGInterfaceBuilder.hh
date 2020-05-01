#ifndef KGINTERFACEBUILDER_HH_
#define KGINTERFACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGCore.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGInterface> KGInterfaceBuilder;

template<> inline bool KGInterfaceBuilder::Begin()
{
    fObject = KGeoBag::KGInterface::GetInstance();
    return true;
}

template<> inline bool KGInterfaceBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "reset") {
        if (aContainer->AsReference<bool>() == true) {
            fObject = KGeoBag::KGInterface::DeleteInstance();
            fObject = KGeoBag::KGInterface::GetInstance();
        }
        return true;
    }
    return false;
}

template<> inline bool KGInterfaceBuilder::AddElement(KContainer* aContainer)
{
    using namespace KGeoBag;

    if (aContainer->Is<KGSurface>()) {
        aContainer->ReleaseTo(fObject, &KGInterface::InstallSurface);
        return true;
    }
    if (aContainer->Is<KGArea>()) {
        KGArea* tArea = nullptr;
        aContainer->ReleaseTo(tArea);
        auto* tSurface = new KGSurface();
        tSurface->SetName(tArea->GetName());
        tSurface->SetTags(tArea->GetTags());
        tSurface->Area(std::shared_ptr<KGArea>(tArea));
        fObject->InstallSurface(tSurface);
        return true;
    }
    if (aContainer->Is<KGSpace>()) {
        aContainer->ReleaseTo(fObject, &KGInterface::InstallSpace);
        return true;
    }
    if (aContainer->Is<KGVolume>()) {
        KGVolume* tVolume = nullptr;
        aContainer->ReleaseTo(tVolume);
        auto* tSpace = new KGSpace();
        tSpace->SetName(tVolume->GetName());
        tSpace->SetTags(tVolume->GetTags());
        tSpace->Volume(std::shared_ptr<KGVolume>(tVolume));
        fObject->InstallSpace(tSpace);
        return true;
    }
    return true;
}

template<> inline bool KGInterfaceBuilder::End()
{
    fObject = nullptr;
    return true;
}

}  // namespace katrin

#endif
