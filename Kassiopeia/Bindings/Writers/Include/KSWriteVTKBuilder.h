#ifndef _Kassiopeia_KSWriteVTKBuilder_h_
#define _Kassiopeia_KSWriteVTKBuilder_h_

#include "KComplexElement.hh"
#include "KSWriteVTK.h"
#include "KThreeVector.hh"
#include "KTwoVector.hh"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSWriteVTK> KSWriteVTKBuilder;

template<> inline bool KSWriteVTKBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "base") {
        aContainer->CopyTo(fObject, &KSWriteVTK::SetBase);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KSWriteVTK::SetPath);
        return true;
    }

    return false;
}

}  // namespace katrin
#endif
