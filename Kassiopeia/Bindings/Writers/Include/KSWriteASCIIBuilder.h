#ifndef Kassiopeia_KSWriteASCIIBuilder_h_
#define Kassiopeia_KSWriteASCIIBuilder_h_

#include "KComplexElement.hh"
#include "KSWriteASCII.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSWriteASCII> KSWriteASCIIBuilder;

template<> inline bool KSWriteASCIIBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "base") {
        aContainer->CopyTo(fObject, &KSWriteASCII::SetBase);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KSWriteASCII::SetPath);
        return true;
    }
    // setting the number of digits of the output values
    if (aContainer->GetName() == "precision") {
        aContainer->CopyTo(fObject, &KSWriteASCII::SetPrecision);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
