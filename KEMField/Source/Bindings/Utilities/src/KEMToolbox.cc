/*
 * KEMToolbox.cc
 *
 *  Created on: 11.05.2015
 *      Author: gosda
 */
#include "KEMToolbox.hh"

#include "KFMMessaging.hh"

using katrin::KContainer;

namespace KEMField
{

void KEMToolbox::AddContainer(KContainer& container, std::string name)
{
    checkKeyIsFree(name);
    KSmartPointer<KContainer> newContainer = container.ReleaseToNewContainer();
    fObjects.insert(NameAndContainer(name, newContainer));
}

bool KEMToolbox::checkKeyIsFree(std::string name)
{
    bool free = (fObjects.find(name) == fObjects.end());
    if (!free) {
        kfmout << "ERROR: Can't have two object in the toolbox with name: " << name << "!" << kfmendl;
        kfmexit(1);
    }
    return free;
}

KSmartPointer<KContainer> KEMToolbox::GetContainer(std::string name)
{
    const auto entry = fObjects.find(name);
    if (entry != fObjects.end())
        return entry->second;
    throw KKeyNotFoundException("KEMToolbox", name, KKeyNotFoundException::noEntry);
}

void KEMToolbox::DeleteAll()
{
    fObjects.clear();
}

}  // namespace KEMField
