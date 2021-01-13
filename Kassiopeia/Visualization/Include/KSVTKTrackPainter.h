#ifndef _Kassiopeia_KSVTKTrackPainter_h_
#define _Kassiopeia_KSVTKTrackPainter_h_

#include "KField.h"
#include "KVTKPainter.h"
#include "KVTKWindow.h"
#include "vtkActor.h"
#include "vtkCellArray.h"
#include "vtkDoubleArray.h"
#include "vtkLookupTable.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkSmartPointer.h"

namespace Kassiopeia
{

class KSVTKTrackPainter : public katrin::KVTKPainter
{
  public:
    KSVTKTrackPainter();
    ~KSVTKTrackPainter() override;

    void Render() override;
    void Display() override;
    void Write() override;

    ;
    K_SET(std::string, File);
    K_SET(std::string, Path);
    K_SET(std::string, OutFile);
    K_SET(std::string, PointObject);
    K_SET(std::string, PointVariable);
    K_SET(std::string, ColorObject);
    K_SET(std::string, ColorVariable);
    K_SET(int, LineWidth)

  private:
    vtkSmartPointer<vtkPoints> fPoints;
    vtkSmartPointer<vtkCellArray> fLines;
    vtkSmartPointer<vtkDoubleArray> fColors;
    vtkSmartPointer<vtkPolyData> fData;
    vtkSmartPointer<vtkPolyDataMapper> fMapper;
    vtkSmartPointer<vtkLookupTable> fTable;
    vtkSmartPointer<vtkActor> fActor;
};

}  // namespace Kassiopeia

#endif
