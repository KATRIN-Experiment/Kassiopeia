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
        aContainer->CopyTo(fObject, &KROOTPad::SetXLow);
        return true;
    }
    if (aContainer->GetName() == "ylow") {
        aContainer->CopyTo(fObject, &KROOTPad::SetYLow);
        return true;
    }
    if (aContainer->GetName() == "xup") {
        aContainer->CopyTo(fObject, &KROOTPad::SetXUp);
        return true;
    }
    if (aContainer->GetName() == "yup") {
        aContainer->CopyTo(fObject, &KROOTPad::SetYUp);
        return true;
    }
    if (aContainer->GetName() == "xmin") {
        aContainer->CopyTo(fObject, &KROOTPad::SetXMin);
        return true;
    }
    if (aContainer->GetName() == "ymin") {
        aContainer->CopyTo(fObject, &KROOTPad::SetYMin);
        return true;
    }
    if (aContainer->GetName() == "xmax") {
        aContainer->CopyTo(fObject, &KROOTPad::SetXMax);
        return true;
    }
    if (aContainer->GetName() == "ymax") {
        aContainer->CopyTo(fObject, &KROOTPad::SetYMax);
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
