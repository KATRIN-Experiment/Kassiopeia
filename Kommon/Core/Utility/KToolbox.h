#ifndef KTOOLBOX_H_
#define KTOOLBOX_H_

#include "KTagged.h"
#include "KSingleton.h"
#include "KInitializationMessage.hh"
#include "KContainer.hh"
#include "KException.h"

#include <string>
#include <set>
#include <map>
#include <vector>
#include <ostream>
#include <memory>
#include <utility>

namespace katrin {

class KToolbox: public KSingleton<KToolbox>
{
public:
    template< class Object = KTagged >
    void Add(Object* ptr, const std::string& tName = "");

    void AddContainer(KContainer& tContainer, std::string tName = "");

    template< class Object = KTagged >
    Object* Get(const std::string& aName = "");

    template<class Object = KTagged >
    std::vector<Object* > GetAll(const std::string& aTag = "");

    void Clear();

protected:
    KToolbox();
    virtual ~KToolbox();

    typedef std::map<std::string,std::shared_ptr<KContainer> > ContainerMap;
    typedef std::set< std::shared_ptr<KContainer> > ContainerSet;
    typedef std::map< std::string, std::shared_ptr<ContainerSet> > TagContainerMap;

    bool CheckKeyIsFree(const std::string& name);

    ContainerMap fObjects;
    TagContainerMap fTagMap;

    friend class KSingleton<KToolbox>;
};

template< class Object>
inline void KToolbox::Add(Object* ptr, const std::string& tName)
{
    KContainer* tContainer = new KContainer();
    tContainer->Set(ptr);

    CheckKeyIsFree(tName);

    //tContainer is emptied in this call
    AddContainer(*tContainer, tName);

    delete tContainer;
    return;
}

inline void KToolbox::AddContainer(KContainer& tContainer, std::string tName) {
    std::shared_ptr<KContainer> tNewContainer(tContainer.ReleaseToNewContainer());

    if (tName == "") {
        if(!(tNewContainer->Is<KNamed>()))
        {
            initmsg(eError) << "No name provided for Object which is not KNamed. This is a Bug" << eom;
        }
        tName = tNewContainer->AsPointer<KNamed>()->GetName();
    }

    CheckKeyIsFree(tName);
    fObjects.insert(ContainerMap::value_type(tName, tNewContainer));

    if ( tNewContainer->Is<KTagged>() ) {
        for (auto tTag : tNewContainer->AsPointer<KTagged>()->GetTags()) {
            auto entry = fTagMap.find(tTag);
            if (entry == fTagMap.end()) {
                std::shared_ptr<ContainerSet> tContainerSet(new ContainerSet());
                fTagMap.insert(std::pair<std::string, std::shared_ptr<ContainerSet> >
                                    (tTag, tContainerSet));
                tContainerSet->insert(tNewContainer);
            }
            else {
                entry->second->insert(tNewContainer);
            }

        }
    }
    return;
}

template< class Object >
inline Object* KToolbox::Get(const std::string& aName)
{
    if (aName == "") {
        for (const auto kvObject : fObjects){
            if ( kvObject.second->AsPointer<Object>() ) {
                return kvObject.second->AsPointer<Object>();
            }
        }
        return nullptr;
    }

    auto entry = fObjects.find(aName);

    if ( (entry != fObjects.end()) && (entry->second->AsPointer<Object>()) )
        return entry->second->AsPointer<Object>();

    initmsg(eWarning) << "No suitable Object called <" << aName << "> in Toolbox" << eom;
    return nullptr;
}

template<class Object>
inline std::vector<Object*> KToolbox::GetAll(const std::string& aTag)
{
    std::vector<Object*> result;

    if (aTag == ""){
        for (const auto kvObject : fObjects){
            if ( kvObject.second->AsPointer<Object>() ) {
                result.push_back(kvObject.second->AsPointer<Object>());
            }
        }
        return result;
    }

    auto entry = fTagMap.find(aTag);
    if (entry == fTagMap.end()) {
        initmsg(eWarning) << "no instances of object with tag <" << aTag << ">" << eom;
        return result;
    }
    for( const auto tContainer : *(entry->second) ) {
        if ( tContainer->AsPointer<Object>() ) {
            result.push_back(tContainer->AsPointer<Object>());
        }
    }
    return result;
}

}

#endif /* KTOOLBOX_H_ */
