/*
 * KEMToolbox.hh
 *
 *  Created on: 11.05.2015
 *      Author: gosda
 */

#ifndef KEMTOOLBOX_HH_
#define KEMTOOLBOX_HH_

#include "KSingleton.h"
#include <string>
#include <map>
#include <vector>

#include "KSmartPointerRelease.hh"
#include "KKeyNotFoundException.hh"

namespace KEMField {

class KEMToolbox: public katrin::KSingleton<KEMToolbox>
{
	friend class KSingleton<KEMToolbox>; // allow calling the constructor

public:
	/*
	 * KEMToolbox assumes ownership of the object.
	 */
	template< class Object >
	void Add(std::string name, Object* ptr);

	/*
	 * KEMToolbox keeps ownership of the object.
	 */
	template< class Object >
	Object* Get(std::string name);

	/*
	 * KEMToolbox keeps ownership of all objects.
	 */
	template< class Object >
	std::vector< std::pair<std::string,Object* > >GetAll();

	/*
	 * KEMToolbox creates own KContainer and releases the object from the given one.
	 */
	void AddContainer( katrin::KContainer& container,std::string name );

	void DeleteAll();

protected:

	KEMToolbox() {}
	virtual ~KEMToolbox() {}

private:
	bool checkKeyIsFree(std::string name);
	KSmartPointer<katrin::KContainer> GetContainer(std::string name);

	typedef std::pair<std::string,KSmartPointer<katrin::KContainer> >
	NameAndContainer;

	typedef std::map<std::string,KSmartPointer<katrin::KContainer> >
	ContainerMap;

	ContainerMap fObjects;

};

template< class Object >
void KEMToolbox::Add(std::string name, Object* ptr)
{
	KSmartPointer<katrin::KContainer> container = new katrin::KContainer();
	container->Set(ptr);
	checkKeyIsFree(name);
	fObjects.insert(NameAndContainer(name,container));
}

template< class Object >
Object* KEMToolbox::Get(std::string name)
{
	KSmartPointer<katrin::KContainer> container = GetContainer(name);
	Object* object = container->AsPointer<Object>();
	if(!object)
		throw KKeyNotFoundException("KEMToolbox",name,KKeyNotFoundException::wrongType);
	return object;
}

template< class Object >
std::vector< std::pair<std::string,Object* > > KEMToolbox::GetAll()
{
    std::vector< std::pair<std::string,Object* > >list;
    for ( auto entry : fObjects )
    {
        Object* candidate = entry.second->AsPointer<Object>();
        if(candidate)
            list.push_back( std::make_pair(entry.first,candidate) );
    }
    return list;
}

} //KEMField
#endif /* KEMTOOLBOX_HH_ */
