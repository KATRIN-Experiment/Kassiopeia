#ifndef KGVTKNEARESTNORMALPAINTER_HH_
#define KGVTKNEARESTNORMALPAINTER_HH_

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

class KGVTKNormalTester : public katrin::KVTKPainter
{
  public:
    KGVTKNormalTester();
    ~KGVTKNormalTester() override;

    void Render() override;
    void Display() override;
    void Write() override;

    void AddSurface(const KGSurface* aSurface);
    void AddSpace(const KGSpace* aSpace);

    K_SET(KThreeVector, SampleDiskOrigin)
    K_SET(KThreeVector, SampleDiskNormal)
    K_SET(double, SampleDiskRadius)
    K_SET(unsigned int, SampleCount)
    K_SET(KGRGBColor, SampleColor)
    K_SET(KGRGBColor, PointColor)
    K_SET(KGRGBColor, NormalColor)
    K_SET(double, NormalLength)
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


inline void KGVTKNormalTester::AddSurface(const KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}

inline void KGVTKNormalTester::AddSpace(const KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}

}  // namespace KGeoBag

#endif
