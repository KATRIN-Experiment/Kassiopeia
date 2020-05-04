#ifndef KGVTKAXIALMESHPAINTER_HH_
#define KGVTKAXIALMESHPAINTER_HH_

#include "KVTKWindow.h"
using katrin::KVTKWindow;

#include "KVTKPainter.h"
using katrin::KVTKPainter;

#include "KGAxialMesh.hh"
#include "KGCore.hh"
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

class KGVTKAxialMeshPainter :
    public KVTKPainter,
    public KGVisitor,
    public KGSurface::Visitor,
    public KGExtendedSurface<KGAxialMesh>::Visitor,
    public KGSpace::Visitor,
    public KGExtendedSpace<KGAxialMesh>::Visitor
{
  public:
    KGVTKAxialMeshPainter();
    virtual ~KGVTKAxialMeshPainter();

  public:
    void Render();
    void Display();
    void Write();

    void VisitSurface(KGSurface* aSurface);
    void VisitExtendedSurface(KGExtendedSurface<KGAxialMesh>* aSurface);
    void VisitSpace(KGSpace* aSpace);
    void VisitExtendedSpace(KGExtendedSpace<KGAxialMesh>* aSpace);

    void SetFile(const std::string& aName);
    const std::string& GetFile() const;

    void SetArcCount(const unsigned int& anArcCount);
    const unsigned int& GetArcCount() const;

    void SetColorMode(const unsigned int& aColorMode);
    const unsigned int& GetColorMode() const;

    static const unsigned int sArea;
    static const unsigned int sAspect;
    static const unsigned int sModulo;

  private:
    void PaintElements();

    KThreeVector fCurrentOrigin;
    KThreeVector fCurrentXAxis;
    KThreeVector fCurrentYAxis;
    KThreeVector fCurrentZAxis;

    KGAxialMeshElementVector* fCurrentElements;

    vtkSmartPointer<vtkLookupTable> fColorTable;
    vtkSmartPointer<vtkDoubleArray> fAreaData;
    vtkSmartPointer<vtkDoubleArray> fAspectData;
    vtkSmartPointer<vtkDoubleArray> fModuloData;
    vtkSmartPointer<vtkPoints> fPoints;
    vtkSmartPointer<vtkCellArray> fLineCells;
    vtkSmartPointer<vtkCellArray> fPolyCells;
    vtkSmartPointer<vtkPolyData> fPolyData;
    vtkSmartPointer<vtkPolyDataMapper> fMapper;
    vtkSmartPointer<vtkActor> fActor;

    std::string fFile;
    unsigned int fArcCount;
    unsigned int fColorMode;
};

}  // namespace KGeoBag

#endif
