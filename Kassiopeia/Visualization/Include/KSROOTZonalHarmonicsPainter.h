#ifndef KSROOTZONALHARMONICSPAINTER_H
#define KSROOTZONALHARMONICSPAINTER_H

#include "KField.h"
#include "KROOTPainter.h"
#include "KROOTWindow.h"
#include <list>

namespace Kassiopeia
{

class KSROOTZonalHarmonicsPainter : public katrin::KROOTPainter
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
    K_SET(std::string, XAxis);
    K_SET(std::string, YAxis);
    K_SET(double, ZMin);
    K_SET(double, ZMax);
    K_SET(double, RMin);
    K_SET(double, RMax);
    K_SET(double, ZDist);
    K_SET(double, RDist);
    K_SET(unsigned, ZMaxSteps);
    K_SET(unsigned, RMaxSteps);
    K_SET(std::string, ElectricFieldName);
    K_SET(std::string, MagneticFieldName);
    K_SET(std::string, File);
    K_SET(std::string, Path);
    K_SET(bool, DrawConvergenceArea);
    K_SET(bool, DrawSourcePoints);
    K_SET(bool, DrawCentralBoundary);
    K_SET(bool, DrawRemoteBoundary);
    //;K_SET( std::string, GeometryType );
    //;K_SET( double, RadialSafetyMargin );

    std::list<std::pair<double,double> > fElCentralConvergenceBounds;
    std::list<std::pair<double,double> > fElRemoteConvergenceBounds;
    std::list<std::pair<double,double> > fElCentralSourcePoints;
    std::list<std::pair<double,double> > fElRemoteSourcePoints;

    std::list<std::pair<double,double> > fMagCentralConvergenceBounds;
    std::list<std::pair<double,double> > fMagRemoteConvergenceBounds;
    std::list<std::pair<double,double> > fMagCentralSourcePoints;
    std::list<std::pair<double,double> > fMagRemoteSourcePoints;
};

}  // namespace Kassiopeia

#endif  // KSROOTZONALHARMONICSPAINTER_H
