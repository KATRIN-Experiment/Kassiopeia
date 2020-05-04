#ifndef _katrin_KROOTPad_h_
#define _katrin_KROOTPad_h_

#include "KField.h"
#include "KPainter.h"
#include "KROOTWindow.h"

#include <vector>
using std::vector;

#include "TCanvas.h"
#include "TH2F.h"
#include "TPad.h"

#include <cstdlib>
#include <iostream>

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

  private:
    typedef vector<KROOTPainter*> PainterVector;
    typedef PainterVector::iterator PainterIt;
    PainterVector fPainters;
    TH2F* fFrame;
    TPad* fPad;
    KROOTWindow* fWindow;

    //settings
    ;
    K_SET(double, xlow);
    ;
    K_SET(double, ylow);
    ;
    K_SET(double, xup);
    ;
    K_SET(double, yup);
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
