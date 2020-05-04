#ifndef _KGeoBag_KGVTKDistanceTester_hh_
#define _KGeoBag_KGVTKDistanceTester_hh_

#include "KVTKWindow.h"
using katrin::KVTKWindow;

#include "KVTKPainter.h"
using katrin::KVTKPainter;

#include "KField.h"
#include "KGCore.hh"
#include "KGRGBColor.hh"
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

class KGVTKDistanceTester : public KVTKPainter
{
  public:
    KGVTKDistanceTester();
    virtual ~KGVTKDistanceTester();

    void Render();
    void Display();
    void Write();

    void AddSurface(const KGSurface* aSurface);
    void AddSpace(const KGSpace* aSpace);

    K_SET(KThreeVector, SampleDiskOrigin)
    K_SET(KThreeVector, SampleDiskNormal)
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
