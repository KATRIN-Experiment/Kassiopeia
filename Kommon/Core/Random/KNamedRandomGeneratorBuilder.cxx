#include "KNamedRandomGeneratorBuilder.h"

#include "KElementProcessor.hh"
#include "KRoot.h"

namespace katrin
{

using SeedType = Kommon::NamedRandomGenerator::result_type;

STATICINT sKommonNamedRandomGeneratorRoot =
    KRootBuilder::ComplexElement<Kommon::NamedRandomGenerator>("NamedRandomGenerator");

STATICINT sKommonNamedRandomGeneratorElement =
    KElementProcessor::ComplexElement<Kommon::NamedRandomGenerator>("NamedRandomGenerator");

STATICINT sKommonNamedRandomGeneratorStructure =
    NamedRandomGeneratorBuilder::Attribute<string>("Name") + NamedRandomGeneratorBuilder::Attribute<SeedType>("Seed");

template<> bool NamedRandomGeneratorBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "Name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "Seed") {
        fObject->SetSeed(aContainer->AsReference<SeedType>());
        return true;
    }
    return false;
}

template<> bool NamedRandomGeneratorBuilder::AddElement(KContainer*)
{
    return false;
}

}  // namespace katrin
