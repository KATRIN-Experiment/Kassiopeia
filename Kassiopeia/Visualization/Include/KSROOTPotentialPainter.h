#ifndef KSROOTPOTENTIALPAINTER_H
#define KSROOTPOTENTIALPAINTER_H

#include "KROOTWindow.h"
using katrin::KROOTWindow;

#include "KROOTPainter.h"
using katrin::KROOTPainter;

#include "KField.h"
#include "KGCore.hh"
#include "KSElectricField.h"

namespace Kassiopeia
{

class KSROOTPotentialPainter : public KROOTPainter
{
  public:
    KSROOTPotentialPainter();
    ~KSROOTPotentialPainter() override;

    void Render() override;
    void Display() override;
    void Write() override;

    double GetXMin() override;
    double GetXMax() override;
    double GetYMin() override;
    double GetYMax() override;

    std::string GetXAxisLabel() override;
    std::string GetYAxisLabel() override;

  public:
    bool CheckPosition(const KThreeVector& aPosition) const;

  private:
    ;
    K_SET(std::string, XAxis);
    ;
    K_SET(std::string, YAxis);
    ;
    K_SET(double, Zmin);
    ;
    K_SET(double, Zmax);
    ;
    K_SET(double, Rmax);
    ;
    K_SET(int, Zsteps);
    ;
    K_SET(int, Rsteps);
    ;
    K_SET(std::string, ElectricFieldName);
    ;
    K_SET(bool, CalcPot);
    TH2D* fMap;
    ;
    K_SET(bool, Comparison);
    ;
    K_SET(std::string, ReferenceFieldName);
};

}  // namespace Kassiopeia

#endif  // KSROOTPOTENTIALPAINTER_H
