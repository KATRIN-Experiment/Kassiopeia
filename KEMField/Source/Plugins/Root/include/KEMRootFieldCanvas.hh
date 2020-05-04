#ifndef KEMROOTFIELDCANVAS_H
#define KEMROOTFIELDCANVAS_H

#include "KEMFieldCanvas.hh"
#include "TBox.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TEllipse.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TLine.h"
#include "TMarker.h"
#include "TStyle.h"

namespace KEMField
{
class KEMRootFieldCanvas : public KEMFieldCanvas
{
  public:
    KEMRootFieldCanvas(double x_1, double x_2, double y_1, double y_2, double zmir = 1.e10, bool isfull = true);
    ~KEMRootFieldCanvas() override;

    void InitializeCanvas();

    void DrawGeomRZ(std::string conicSectfile, std::string wirefile, std::string coilfile);

    void DrawGeomXY(double z, std::string conicSectfile, std::string wirefile, std::string coilfile);

    void DrawFieldMapCube(double x_1, double x_2, double y_1, double y_2);

    void DrawFieldMap(std::vector<double> x, std::vector<double> y, std::vector<double> V, bool xy = false,
                      double z = 0) override;
    void DrawComparisonMap(int nPoints, std::vector<double> x, std::vector<double> y, std::vector<double> V1,
                           std::vector<double> V2) override;
    void DrawFieldLines(std::vector<double> x, std::vector<double> y) override;
    void LabelAxes(std::string xname, std::string yname, std::string zname) override;
    void LabelCanvas(std::string title) override;
    void SaveAs(std::string savename) override;

  private:
    TCanvas* canvas;
    TH2F* hist;
};
}  // namespace KEMField

#endif
