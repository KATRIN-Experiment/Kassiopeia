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

    void DrawGeomRZ(const std::string& conicSectfile, const std::string& wirefile, const std::string& coilfile);

    static void DrawGeomXY(double z, const std::string& conicSectfile, const std::string& wirefile,
                           const std::string& coilfile);

    void DrawFieldMapCube(double x_1, double x_2, double y_1, double y_2);

    void DrawFieldMap(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& V,
                      bool xy = false, double z = 0) override;
    void DrawComparisonMap(int nPoints, const std::vector<double>& x, const std::vector<double>& y,
                           const std::vector<double>& V1, const std::vector<double>& V2) override;
    void DrawFieldLines(const std::vector<double>& x, const std::vector<double>& y) override;
    void LabelAxes(const std::string& xname, const std::string& yname, const std::string& zname) override;
    void LabelCanvas(const std::string& title) override;
    void SaveAs(const std::string& savename) override;

  private:
    TCanvas* canvas;
    TH2F* hist;
};
}  // namespace KEMField

#endif
