/*
 * KToolbox.cxx
 *
 *  Created on: 16.11.2011
 *      Author: Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#include "KToolbox.h"

using namespace std;

namespace katrin
{

KToolbox::KToolbox() : fObjects(), fTagMap()
{
    fTagMap.insert(TagContainerMap::value_type(string(""), make_shared<ContainerSet>()));
};

KToolbox::~KToolbox()
{
    Clear();
}

void KToolbox::Clear()
{
    fObjects.clear();
    fTagMap.clear();
};

bool KToolbox::CheckKeyIsFree(const string& aName) const
{
    if (HasKey(aName)) {
        initmsg(eError) << "Multiple instances of object with name <" << aName << ">." << eom;
    }
    return true;
}

std::shared_ptr<KContainer> KToolbox::Remove(const std::string& aName)
{
    initmsg(eDebug) << "Removing object <" << aName << "> from Toolbox: " << this << eom;

    auto entry = fObjects.find(aName);

    if (entry != fObjects.end()) {
        auto ptr = entry->second;
        fObjects.erase(entry);
        return ptr;
    }

    return nullptr;
}

void KToolbox::AddContainer(KContainer& aContainer, const std::string& aName)
{
    ContainerPtr tNewContainer(aContainer.ReleaseToNewContainer());

    std::string tName = aName;
    if (tName == "") {
        if (!tNewContainer->Is<KNamed>()) {
            initmsg(eError) << "No name provided for Object which is not KNamed. This is a Bug." << eom;
            return;
        }
        tName = tNewContainer->AsPointer<KNamed>()->GetName();
    }

    initmsg(eDebug) << "Adding container <" << tName << "> to Toolbox: " << this << eom;

    CheckKeyIsFree(tName);
    fObjects.insert(ContainerMap::value_type(tName, tNewContainer));

    if (tNewContainer->Is<KTagged>()) {
        for (auto tTag : tNewContainer->AsPointer<KTagged>()->GetTags()) {
            auto entry = fTagMap.find(tTag);
            if (entry == fTagMap.end()) {
                auto tContainerSet = std::make_shared<ContainerSet>();
                fTagMap.insert(TagContainerMap::value_type(tTag, tContainerSet));
                tContainerSet->insert(tNewContainer);
            }
            else {
                entry->second->insert(tNewContainer);
            }
        }
    }
    return;
}

} /* namespace katrin */
