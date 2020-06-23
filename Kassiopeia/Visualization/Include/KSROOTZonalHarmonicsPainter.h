#ifndef KSROOTZONALHARMONICSPAINTER_H
#define KSROOTZONALHARMONICSPAINTER_H

#include "KROOTWindow.h"
using katrin::KROOTWindow;

#include "KROOTPainter.h"
using katrin::KROOTPainter;

#include "KField.h"
#include "TGraph.h"
#include "TPolyMarker.h"

namespace Kassiopeia
{

class KSROOTZonalHarmonicsPainter : public KROOTPainter
{
  public:
    KSROOTZonalHarmonicsPainter();
    ~KSROOTZonalHarmonicsPainter() override;

    void Render() override;
    void Display() override;
    void Write() override;

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
    K_SET(double, Rmin);
    ;
    K_SET(double, Rmax);
    ;
    K_SET(double, Zdist);
    ;
    K_SET(double, Rdist);
    ;
    K_SET(unsigned, ZMaxSteps);
    ;
    K_SET(unsigned, RMaxSteps);
    ;
    K_SET(std::string, ElectricFieldName);
    ;
    K_SET(std::string, MagneticFieldName);
    ;
    K_SET(std::string, File);
    ;
    K_SET(std::string, Path);
    ;
    K_SET(bool, DrawSourcePoints);
    ;
    K_SET(bool, DrawSourcePointArea);
    //;K_SET( std::string, GeometryType );
    //;K_SET( double, RadialSafetyMargin );

    //std::vector<std::pair<double,double> > fZRPoints;

    // TODO: use TPolyLine
    TGraph* fElZHConvergenceGraph;
    TGraph* fElZHCentralGraph;
    TGraph* fElZHRemoteGraph;

    TGraph* fMagZHConvergenceGraph;
    TGraph* fMagZHCentralGraph;
    TGraph* fMagZHRemoteGraph;

    TPolyMarker* fElZHPoints;
    TPolyMarker* fMagZHPoints;
};

}  // namespace Kassiopeia

#endif  // KSROOTZONALHARMONICSPAINTER_H
