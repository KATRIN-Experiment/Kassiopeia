#ifndef _KGeoBag_KGVTKDistanceTester_hh_
#define _KGeoBag_KGVTKDistanceTester_hh_

#include "KField.h"
#include "KGCore.hh"
#include "KGRGBColor.hh"
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

namespace KGeoBag
{

class KGVTKDistanceTester : public katrin::KVTKPainter
{
  public:
    KGVTKDistanceTester();
    ~KGVTKDistanceTester() override;

    void Render() override;
    void Display() override;
    void Write() override;

    void AddSurface(const KGSurface* aSurface);
    void AddSpace(const KGSpace* aSpace);

    K_SET(katrin::KThreeVector, SampleDiskOrigin)
    K_SET(katrin::KThreeVector, SampleDiskNormal)
    K_SET(double, SampleDiskRadius)
    K_SET(unsigned int, SampleCount)
    K_SET(double, VertexSize)

  private:
    std::vector<const KGSurface*> fSurfaces;
    std::vector<const KGSpace*> fSpaces;

    vtkSmartPointer<vtkPoints> fPoints;
    vtkSmartPointer<vtkDoubleArray> fValues;
    vtkSmartPointer<vtkCellArray> fCells;
    vtkSmartPointer<vtkPolyData> fPolyData;
    vtkSmartPointer<vtkLookupTable> fTable;
    vtkSmartPointer<vtkPolyDataMapper> fMapper;
    vtkSmartPointer<vtkActor> fActor;
};

inline void KGVTKDistanceTester::AddSurface(const KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}

inline void KGVTKDistanceTester::AddSpace(const KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}

}  // namespace KGeoBag

#endif
