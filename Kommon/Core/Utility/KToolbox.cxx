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

KToolbox::KToolbox() :
    fObjects(),
    fTagMap()
{
    fTagMap.insert( TagContainerMap::value_type(
        string( "" ), make_shared<ContainerSet>()
    ) );
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

bool KToolbox::CheckKeyIsFree(const string& name)
{
    if( fObjects.find(name) != fObjects.end())
    {
        initmsg(eError) << "Multiple instances of object with name <" << name << ">." << eom;
    }
    return true;
}

} /* namespace katrin */
