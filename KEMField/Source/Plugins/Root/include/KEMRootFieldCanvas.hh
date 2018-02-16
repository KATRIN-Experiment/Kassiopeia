#ifndef KEMROOTFIELDCANVAS_H
#define KEMROOTFIELDCANVAS_H

#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TMarker.h"
#include "TLine.h"
#include "TBox.h"
#include "TEllipse.h"
#include "TColor.h"
#include "TStyle.h"

#include "KEMFieldCanvas.hh"

namespace KEMField
{
  class KEMRootFieldCanvas : public KEMFieldCanvas
  {
  public:
    KEMRootFieldCanvas(double x_1,
		       double x_2,
		       double y_1,
		       double y_2,
		       double zmir = 1.e10,
		       bool   isfull = true);
    virtual ~KEMRootFieldCanvas();

    void InitializeCanvas();

    void DrawGeomRZ(std::string conicSectfile,
		    std::string wirefile,
		    std::string coilfile);

    void DrawGeomXY(double    z,
		    std::string conicSectfile,
		    std::string wirefile,
		    std::string coilfile);

    void DrawFieldMapCube(double x_1,
			  double x_2,
			  double y_1,
			  double y_2);

    void DrawFieldMap(std::vector<double> x,
		      std::vector<double> y,
		      std::vector<double> V,
		      bool xy = false,
		      double z = 0);
    void DrawComparisonMap(int nPoints,
			   std::vector<double> x,
			   std::vector<double> y,
			   std::vector<double> V1,
			   std::vector<double> V2);
    void DrawFieldLines(std::vector<double> x,std::vector<double> y);
    void LabelAxes(std::string xname,std::string yname,std::string zname);
    void LabelCanvas(std::string title);
    void SaveAs(std::string savename);

  private:
    TCanvas* canvas;
    TH2F* hist;
  };
}

#endif
