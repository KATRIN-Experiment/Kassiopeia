#ifndef _katrin_KROOTPad_h_
#define _katrin_KROOTPad_h_

#include "KField.h"
#include "KPainter.h"
#include "KROOTWindow.h"
#include "TCanvas.h"
#include "TH2F.h"
#include "TPad.h"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace katrin
{

class KROOTPainter;

class KROOTPad : public KROOTWindow
{

  public:
    KROOTPad();
    ~KROOTPad() override;

    void Render() override;
    void Display() override;
    void Write() override;

    void AddPainter(KPainter* aPainter) override;
    void RemovePainter(KPainter* aPainter) override;

    void SetWindow(KWindow* aWindow);
    void ClearWindow(KWindow* aWindow);

    TPad* GetPad();
    KROOTWindow* GetWindow();

  private:
    typedef std::vector<KROOTPainter*> PainterVector;
    using PainterIt = PainterVector::iterator;
    PainterVector fPainters;
    TH2F* fFrame;
    TPad* fPad;
    KROOTWindow* fWindow;

    //settings
    K_SET_GET(double, XLow);
    K_SET_GET(double, YLow);
    K_SET_GET(double, XUp);
    K_SET_GET(double, YUp);
};


}  // namespace katrin

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////


#endif
