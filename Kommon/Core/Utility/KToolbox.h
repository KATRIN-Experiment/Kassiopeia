#ifndef KTOOLBOX_H_
#define KTOOLBOX_H_

#include "KContainer.hh"
#include "KException.h"
#include "KInitializationMessage.hh"
#include "KSingleton.h"
#include "KTagged.h"

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace katrin
{

class KToolbox : public KSingleton<KToolbox>
{
  public:
    template<class Object = KTagged> void Add(Object* anObject, const std::string& aName = "");

    template<class Object = KTagged> void Replace(Object* anObject, const std::string& aName = "");

    void AddContainer(KContainer& aContainer, const std::string& aName = "");

    std::shared_ptr<KContainer> Remove(const std::string& aName = "");

    template<class Object = KTagged> Object* Get(const std::string& aName = "") const;

    template<class Object = KTagged> std::vector<Object*> GetAll(const std::string& aTag = "") const;

    template<class Object = KTagged> std::vector<std::string> FindAll() const;

    void Clear();

    inline bool HasKey(const std::string& aName) const
    {
        return fObjects.find(aName) != fObjects.end();
    }

    inline bool HasTag(const std::string& aTag) const
    {
        return fTagMap.find(aTag) != fTagMap.end();
    }

  protected:
    typedef std::shared_ptr<KContainer> ContainerPtr;
    using ContainerMap = std::map<std::string, ContainerPtr>;
    using ContainerSet = std::set<ContainerPtr>;
    using TagContainerMap = std::map<std::string, std::shared_ptr<ContainerSet>>;

    KToolbox();
    ~KToolbox() override;

    bool CheckKeyIsFree(const std::string& aName) const;

  private:
    ContainerMap fObjects;
    TagContainerMap fTagMap;

    friend class KSingleton<KToolbox>;
};

template<class Object> inline void KToolbox::Add(Object* anObject, const std::string& aName)
{
    initmsg(eDebug) << "Adding object <" << aName << "> to Toolbox" << eom;

    CheckKeyIsFree(aName);

    auto* tContainer = new KContainer();
    tContainer->Set(anObject);

    //tContainer is emptied in this call
    AddContainer(*tContainer, aName);

    delete tContainer;
    return;
}

template<class Object> inline void KToolbox::Replace(Object* anObject, const std::string& aName)
{
    if (aName == "")
        return;

    initmsg(eDebug) << "Replacing object <" << aName << "> in Toolbox" << eom;

    auto entry = fObjects.find(aName);

    if (entry != fObjects.end()) {
        entry->second->Set(anObject);
        return;
    }

    initmsg(eError) << "No suitable Object called <" << aName << "> in Toolbox" << eom;
    return;
}

template<class Object> inline Object* KToolbox::Get(const std::string& aName) const
{
    if (aName == "") {
        for (const auto& kvObject : fObjects) {
            if (kvObject.second->AsPointer<Object>()) {
                return kvObject.second->AsPointer<Object>();
            }
        }
        return nullptr;
    }

    initmsg(eDebug) << "Getting object <" << aName << "> from Toolbox" << eom;

    auto entry = fObjects.find(aName);

    if ((entry != fObjects.end()) && entry->second->AsPointer<Object>()) {
        return entry->second->AsPointer<Object>();
    }

    initmsg(eDebug) << "No suitable Object called <" << aName << "> in Toolbox" << eom;
    return nullptr;
}

template<class Object> inline std::vector<Object*> KToolbox::GetAll(const std::string& aTag) const
{
    std::vector<Object*> result;

    if (aTag == "") {
        for (const auto& kvObject : fObjects) {
            if (kvObject.second->AsPointer<Object>()) {
                result.push_back(kvObject.second->AsPointer<Object>());
            }
        }
        return result;
    }

    initmsg(eDebug) << "Getting objects tagged <" << aTag << "> from Toolbox" << eom;

    auto entry = fTagMap.find(aTag);

    if (entry != fTagMap.end()) {
        for (const auto& tContainer : *(entry->second)) {
            if (tContainer->AsPointer<Object>()) {
                result.push_back(tContainer->AsPointer<Object>());
            }
        }
        return result;
    }

    initmsg(eDebug) << "No instances of object with tag <" << aTag << "> in Toolbox" << eom;
    return result;
}

template<class Object> inline std::vector<std::string> KToolbox::FindAll() const
{
    std::vector<std::string> result;

    for (const auto& kvObject : fObjects) {
        if (kvObject.second->AsPointer<Object>()) {
            result.push_back(kvObject.first);
        }
    }
    return result;
}

}  // namespace katrin

#endif /* KTOOLBOX_H_ */
