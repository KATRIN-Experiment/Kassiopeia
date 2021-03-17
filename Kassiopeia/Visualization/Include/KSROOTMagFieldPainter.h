#ifndef KSROOTMAGFIELDPAINTER_H
#define KSROOTMAGFIELDPAINTER_H

#include "KField.h"
#include "KROOTPainter.h"
#include "KROOTWindow.h"
#include "KSMagneticField.h"

namespace Kassiopeia
{

class KSROOTMagFieldPainter : public katrin::KROOTPainter
{
  public:
    KSROOTMagFieldPainter();
    ~KSROOTMagFieldPainter() override;

    void Render() override;
    void Display() override;
    void Write() override;
    virtual void FieldMapX(KSMagneticField* tMagField, double tDeltaZ, double tDeltaR);
    virtual void FieldMapZ(KSMagneticField* tMagField, double tDeltaZ, double tDeltaR);

    double GetXMin() override;
    double GetXMax() override;
    double GetYMin() override;
    double GetYMax() override;

    std::string GetXAxisLabel() override;
    std::string GetYAxisLabel() override;

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
    K_SET(double, Zfix);
    ;
    K_SET(double, Rmax);
    ;
    K_SET(int, Zsteps);
    ;
    K_SET(int, Rsteps);
    ;
    K_SET(std::string, MagneticFieldName);
    ;
    K_SET(bool, AxialSymmetry);
    ;
    K_SET(std::string, Plot);
    ;
    K_SET(bool, UseLogZ);
    ;
    K_SET(bool, GradNumerical);
    ;
    K_SET(std::string, Draw);
    TH2D* fMap;
};

}  // namespace Kassiopeia

#endif  // KSROOTMAGFIELDPAINTER_H
