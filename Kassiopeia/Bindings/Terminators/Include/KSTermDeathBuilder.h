#ifndef Kassiopeia_KSTermDeathBuilder_h_
#define Kassiopeia_KSTermDeathBuilder_h_

#include "KComplexElement.hh"
#include "KSTermDeath.h"

using namespace Kassiopeia;
namespace katrin
{

//! this terminator kills everything it sees.
//! it is usually used in conjunction with a geometric piece to kill particles when they enter a space or strike a surface.
//! @example <ksterm_death name="term_death"/>
typedef KComplexElement<KSTermDeath> KSTermDeathBuilder;

template<> inline bool KSTermDeathBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
