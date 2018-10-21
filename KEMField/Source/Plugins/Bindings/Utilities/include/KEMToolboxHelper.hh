/*
 * KEMToolboxHelper.hh
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_UTILITIES_INCLUDE_KEMTOOLBOXHELPER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_UTILITIES_INCLUDE_KEMTOOLBOXHELPER_HH_

#include "KContainer.hh"
#include "KEMToolbox.hh"

namespace katrin {

template< typename ObjectType >
/** Put the object in the KContainer into the KEMFieldToolbox if it is of
 * the correct type (or a subtype).
 * Return true if the type is correct and false otherwise.
 */
bool tryLoadAs(KContainer* aContainer){
    if(aContainer->Is<ObjectType>()) {
        ObjectType* object;
        aContainer->ReleaseTo(object);
        std::string name = aContainer->GetName();
        KEMField::KEMToolbox::GetInstance().Add<ObjectType>(name,object);
        return true;
    }
    else return false;
}

} /* namespace katrin */


#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_UTILITIES_INCLUDE_KEMTOOLBOXHELPER_HH_ */
