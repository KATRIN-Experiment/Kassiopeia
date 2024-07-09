//
// Created by trost on 25.07.16.
//

#ifndef KASPER_KROOTWINDOWBUILDER_H_H
#define KASPER_KROOTWINDOWBUILDER_H_H
#include "KComplexElement.hh"
#include "KROOTPainter.h"
#include "KROOTWindow.h"

namespace katrin
{

typedef KComplexElement<KROOTWindow> KROOTWindowBuilder;

template<> inline bool KROOTWindowBuilder::Begin()
{
    fObject = new KROOTWindow();
    return true;
}

template<> inline bool KROOTWindowBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "canvas_width") {
        aContainer->CopyTo(fObject, &KROOTWindow::SetCanvasWidth);
        return true;
    }
    if (aContainer->GetName() == "canvas_height") {
        aContainer->CopyTo(fObject, &KROOTWindow::SetCanvasHeight);
        return true;
    }
    if (aContainer->GetName() == "active") {
        aContainer->CopyTo(fObject, &KROOTWindow::SetActive);
        return true;
    }
    if (aContainer->GetName() == "write_enabled") {
        aContainer->CopyTo(fObject, &KROOTWindow::SetWriteEnabled);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KROOTWindow::SetPath);
        return true;
    }
    if (aContainer->GetName() == "xmin") {
        aContainer->CopyTo(fObject, &KROOTWindow::SetXMin);
        return true;
    }
    if (aContainer->GetName() == "ymin") {
        aContainer->CopyTo(fObject, &KROOTWindow::SetYMin);
        return true;
    }
    if (aContainer->GetName() == "xmax") {
        aContainer->CopyTo(fObject, &KROOTWindow::SetXMax);
        return true;
    }
    if (aContainer->GetName() == "ymax") {
        aContainer->CopyTo(fObject, &KROOTWindow::SetYMax);
        return true;
    }
    return false;
}

template<> inline bool KROOTWindowBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KPainter>() == true) {
        aContainer->ReleaseTo(fObject, &KROOTWindow::AddPainter);
        return true;
    }
    if (aContainer->Is<KWindow>() == true) {
        aContainer->ReleaseTo(fObject, &KROOTWindow::AddWindow);
        return true;
    }
    return false;
}

template<> inline bool KROOTWindowBuilder::End()
{
    fObject->Render();
    fObject->Display();
    fObject->Write();
    delete fObject;
    return true;
}

}  // namespace katrin

#endif  //KASPER_KROOTWINDOWBUILDER_H_H
