#ifndef KEMVTKFIELDCANVAS_H
#define KEMVTKFIELDCANVAS_H

#include "KEMFieldCanvas.hh"
#include "vtkAxis.h"
#include "vtkChartHistogram2D.h"
#include "vtkChartLegend.h"
#include "vtkColorTransferFunction.h"
#include "vtkContextScene.h"
#include "vtkContextView.h"
#include "vtkImageData.h"
#include "vtkMath.h"
#include "vtkPNGWriter.h"
#include "vtkPlotHistogram2D.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkSmartPointer.h"
#include "vtkWindowToImageFilter.h"

namespace KEMField
{
class KEMVTKFieldCanvas : public KEMFieldCanvas
{
  public:
    KEMVTKFieldCanvas(double x_1, double x_2, double y_1, double y_2, double zmir = 1.e10, bool isfull = true);
    virtual ~KEMVTKFieldCanvas();

    void InitializeCanvas();

    void DrawFieldMap(std::vector<double> x, std::vector<double> y, std::vector<double> V, bool xy = false,
                      double z = 0);
    void DrawComparisonMap(int nPoints, std::vector<double> x, std::vector<double> y, std::vector<double> V1,
                           std::vector<double> V2);
    void DrawFieldLines(std::vector<double> x, std::vector<double> y);
    void LabelAxes(std::string xname, std::string yname, std::string zname);
    void LabelCanvas(std::string title);
    void SaveAs(std::string savename);

  private:
    vtkSmartPointer<vtkContextView> view;
    vtkSmartPointer<vtkChartHistogram2D> chart;
    vtkSmartPointer<vtkImageData> data;
    vtkSmartPointer<vtkColorTransferFunction> transferFunction;
};
}  // namespace KEMField

#endif
