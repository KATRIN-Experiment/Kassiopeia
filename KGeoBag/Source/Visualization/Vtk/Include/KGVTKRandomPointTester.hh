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

class KGVTKRandomPointTester : public katrin::KVTKPainter
{
  public:
    KGVTKRandomPointTester();
    ~KGVTKRandomPointTester() override;

    void Render() override;
    void Display() override;
    void Write() override;

    void AddSurface(const KGSurface* aSurface);
    void AddSpace(const KGSpace* aSpace);

    K_SET(KGRGBColor, SampleColor)
    K_SET(double, VertexSize)
    K_SET(std::vector<katrin::KThreeVector*>, SamplePoints)

  private:
    std::vector<const KGSurface*> fSurfaces;
    std::vector<const KGSpace*> fSpaces;

    vtkSmartPointer<vtkPoints> fPoints;
    vtkSmartPointer<vtkUnsignedCharArray> fColors;
    vtkSmartPointer<vtkCellArray> fCells;
    vtkSmartPointer<vtkPolyData> fPolyData;
    vtkSmartPointer<vtkPolyDataMapper> fMapper;
    vtkSmartPointer<vtkActor> fActor;
};


inline void KGVTKRandomPointTester::AddSurface(const KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}

inline void KGVTKRandomPointTester::AddSpace(const KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}

}  // namespace KGeoBag

#endif
