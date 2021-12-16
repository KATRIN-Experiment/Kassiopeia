#ifndef KEMVTKFIELDCANVAS_H
#define KEMVTKFIELDCANVAS_H

#include "KEMFieldCanvas.hh"

#include "vtkChartXY.h"
#include "vtkColorTransferFunction.h"
#include "vtkContextView.h"
#include "vtkImageData.h"

namespace KEMField
{
class KEMVTKFieldCanvas : public KEMFieldCanvas
{
  public:
    KEMVTKFieldCanvas(double x_1, double x_2, double y_1, double y_2, double zmir = 1.e10, bool isfull = true);
    ~KEMVTKFieldCanvas() override;

    void InitializeCanvas();

    void DrawFieldGraph(const std::vector<double>& x, const std::vector<double>& V) /*override*/;
    void DrawFieldMap(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& V,
                      bool xy = false, double z = 0) override;
    void DrawComparisonMap(int nPoints, const std::vector<double>& x, const std::vector<double>& y,
                           const std::vector<double>& V1, const std::vector<double>& V2) override;
    void DrawFieldLines(const std::vector<double>& x, const std::vector<double>& y) override;
    void LabelAxes(const std::string& xname, const std::string& yname, const std::string& zname = "") override;
    void LabelCanvas(const std::string& title) override;
    void SaveAs(const std::string& savename) override;
    void Export(const std::string& savename);
    void View();

  private:
    vtkSmartPointer<vtkContextView> view;
    vtkSmartPointer<vtkChartXY> chart;
    vtkSmartPointer<vtkImageData> data;
    vtkSmartPointer<vtkColorTransferFunction> transferFunction;
};
}  // namespace KEMField

#endif
