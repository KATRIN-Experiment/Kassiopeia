#ifndef Kassiopeia_KSGenGeneratorTextFileBuilder_h_
#define Kassiopeia_KSGenGeneratorTextFileBuilder_h_

#include "KComplexElement.hh"
#include "KSGenGeneratorTextFile.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenGeneratorTextFile> KSGenGeneratorTextFileBuilder;

template<> inline bool KSGenGeneratorTextFileBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "base") {
        aContainer->CopyTo(fObject, &KSGenGeneratorTextFile::SetBase);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KSGenGeneratorTextFile::SetPath);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
