#ifndef _KGeoBag_KGVTKOutsideTester_hh_
#define _KGeoBag_KGVTKOutsideTester_hh_

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

class KGVTKOutsideTester : public katrin::KVTKPainter
{
  public:
    KGVTKOutsideTester();
    ~KGVTKOutsideTester() override;

    void Render() override;
    void Display() override;
    void Write() override;

    void AddSurface(const KGSurface* aSurface);
    void AddSpace(const KGSpace* aSpace);

    K_SET(katrin::KThreeVector, SampleDiskOrigin)
    K_SET(katrin::KThreeVector, SampleDiskNormal)
    K_SET(double, SampleDiskRadius)
    K_SET(unsigned int, SampleCount)
    K_SET(KGRGBColor, InsideColor)
    K_SET(KGRGBColor, OutsideColor)
    K_SET(double, VertexSize)

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

inline void KGVTKOutsideTester::AddSurface(const KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}

inline void KGVTKOutsideTester::AddSpace(const KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}

}  // namespace KGeoBag

#endif
