#ifndef _katrin_KROOTWindow_h_
#define _katrin_KROOTWindow_h_

#include "KField.h"
#include "KFile.h"
#include "KWindow.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TH2F.h"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace katrin
{

class KROOTPainter;
class KROOTPad;

class KROOTWindow : public KWindow
{

  public:
    KROOTWindow();
    ~KROOTWindow() override;

    void Render() override;
    void Display() override;
    void Write() override;

    void AddPainter(KPainter* aPainter) override;
    void RemovePainter(KPainter* aPainter) override;

    void AddWindow(KWindow* aWindow);
    void RemoveWindow(KWindow* aWindow);

    TPad* GetPad();

  private:
    typedef std::vector<KROOTPainter*> PainterVector;
    using PainterIt = PainterVector::iterator;
    PainterVector fPainters;
    using PadVector = std::vector<KROOTPad*>;
    using PadIt = PadVector::iterator;
    PadVector fPads;
    TH2F* fFrame;
    TApplication* fApplication;
    TCanvas* fCanvas;

    //settings
    ;
    K_SET(unsigned int, CanvasWidth);
    ;
    K_SET(unsigned int, CanvasHeight);
    ;
    K_SET(bool, Active);
    ;
    K_SET(bool, WriteEnabled);
    ;
    K_SET(std::string, Path);
};


}  // namespace katrin

#endif
