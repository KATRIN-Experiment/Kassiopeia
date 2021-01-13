#ifndef _Kassiopeia_KSROOTTrackPainter_h_
#define _Kassiopeia_KSROOTTrackPainter_h_

#include "KField.h"
#include "KROOTPainter.h"
#include "KROOTWindow.h"
#include "KSVisualizationMessage.h"
#include "TColor.h"
#include "TMultiGraph.h"

namespace Kassiopeia
{
class KSReadFileROOT;

class KSROOTTrackPainter : public katrin::KROOTPainter
{
  public:
    KSROOTTrackPainter();
    ~KSROOTTrackPainter() override;

    void Render() override;
    void Display() override;
    void Write() override;

    double GetXMin() override;
    double GetXMax() override;
    double GetYMin() override;
    double GetYMax() override;

    std::string GetXAxisLabel() override;
    std::string GetYAxisLabel() override;

    void AddBaseColor(TColor aColor, double aFraction);

    typedef enum  // NOLINT(modernize-use-using)
    {
        eColorFix,
        eColorStep,
        eColorTrack
    } ColorMode;

    typedef enum  // NOLINT(modernize-use-using)
    {
        eColorFPDRings,
        eColorDefault,
        eColorCustom
    } ColorPalette;

    typedef enum  // NOLINT(modernize-use-using)
    {
        ePlotStep,
        ePlotTrack
    } PlotMode;

  private:
    void CreateColors(KSReadFileROOT& aReader);

  private:
    ;
    K_SET(std::string, Path);
    ;
    K_SET(std::string, Base);
    ;
    K_SET(std::string, XAxis);
    ;
    K_SET(std::string, YAxis);
    ;
    K_SET(std::string, StepOutputGroupName);
    ;
    K_SET(std::string, PositionName);
    ;
    K_SET(std::string, TrackOutputGroupName);
    ;
    K_SET(std::string, ColorVariable);
    ;
    K_SET(ColorMode, ColorMode);
    ;
    K_SET(ColorPalette, ColorPalette);
    ;
    K_SET(std::string, DrawOptions);
    ;
    K_SET(PlotMode, PlotMode);
    ;
    K_SET(bool, AxialMirror);
    TMultiGraph* fMultigraph;
    std::vector<std::pair<TColor, double>> fBaseColors;
    std::vector<Color_t> fColorVector;
};

inline void KSROOTTrackPainter::AddBaseColor(TColor aColor, double aFraction = -1.0)
{
    vismsg(eNormal) << "ROOTTrackPainter adding color " << aColor.GetRed() << "," << aColor.GetGreen() << ","
                    << aColor.GetBlue() << " with fraction " << aFraction << eom;
    fBaseColors.emplace_back(aColor, aFraction);
}

}  // namespace Kassiopeia

#endif
