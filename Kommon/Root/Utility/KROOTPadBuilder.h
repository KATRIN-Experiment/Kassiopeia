//
// Created by trost on 25.07.16.
//

#ifndef KASPER_KROOTPADBUILDER_H
#define KASPER_KROOTPADBUILDER_H
#include "KComplexElement.hh"
#include "KROOTPad.h"

namespace katrin
{

typedef KComplexElement<KROOTPad> KROOTPadBuilder;

template<> inline bool KROOTPadBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "xlow") {
        aContainer->CopyTo(fObject, &KROOTPad::Setxlow);
        return true;
    }
    if (aContainer->GetName() == "ylow") {
        aContainer->CopyTo(fObject, &KROOTPad::Setylow);
        return true;
    }
    if (aContainer->GetName() == "xup") {
        aContainer->CopyTo(fObject, &KROOTPad::Setxup);
        return true;
    }
    if (aContainer->GetName() == "yup") {
        aContainer->CopyTo(fObject, &KROOTPad::Setyup);
        return true;
    }
    return false;
}

template<> inline bool KROOTPadBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KPainter>() == true) {
        aContainer->ReleaseTo(fObject, &KROOTPad::AddPainter);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif  //KASPER_KROOTPADBUILDER_H
