#ifndef _KGeoBag_KGVTKPointTester_hh_
#define _KGeoBag_KGVTKPointTester_hh_

#include "KField.h"
#include "KGCore.hh"
#include "KGRGBColor.hh"
#include "KVTKPainter.h"
#include "KVTKWindow.h"
#include "vtkActor.h"
#include "vtkCellArray.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkSmartPointer.h"
#include "vtkUnsignedCharArray.h"

namespace KGeoBag
{

class KGVTKPointTester : public katrin::KVTKPainter
{
  public:
    KGVTKPointTester();
    ~KGVTKPointTester() override;

    void Render() override;
    void Display() override;
    void Write() override;

    void AddSurface(const KGSurface* aSurface);
    void AddSpace(const KGSpace* aSpace);

    K_SET(katrin::KThreeVector, SampleDiskOrigin)
    K_SET(katrin::KThreeVector, SampleDiskNormal)
    K_SET(double, SampleDiskRadius)
    K_SET(unsigned int, SampleCount)
    K_SET(KGRGBColor, SampleColor)
    K_SET(KGRGBColor, PointColor)
    K_SET(double, VertexSize)
    K_SET(double, LineSize)

  private:
    std::vector<const KGSurface*> fSurfaces;
    std::vector<const KGSpace*> fSpaces;

    vtkSmartPointer<vtkPoints> fPoints;
    vtkSmartPointer<vtkUnsignedCharArray> fColors;
    vtkSmartPointer<vtkCellArray> fPointCells;
    vtkSmartPointer<vtkCellArray> fLineCells;
    vtkSmartPointer<vtkPolyData> fPolyData;
    vtkSmartPointer<vtkPolyDataMapper> fMapper;
    vtkSmartPointer<vtkActor> fActor;
};

inline void KGVTKPointTester::AddSurface(const KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}

inline void KGVTKPointTester::AddSpace(const KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}

}  // namespace KGeoBag

#endif
