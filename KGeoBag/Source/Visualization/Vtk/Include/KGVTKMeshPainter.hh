#ifndef KGVTKMESHPAINTER_HH_
#define KGVTKMESHPAINTER_HH_

#include "KGCore.hh"
#include "KGMesh.hh"
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

class KGVTKMeshPainter :
    public katrin::KVTKPainter,
    public KGVisitor,
    public KGSurface::Visitor,
    public KGExtendedSurface<KGMesh>::Visitor,
    public KGSpace::Visitor,
    public KGExtendedSpace<KGMesh>::Visitor
{
  public:
    KGVTKMeshPainter();
    ~KGVTKMeshPainter() override;

  public:
    void Render() override;
    void Display() override;
    void Write() override;

    void VisitSurface(KGSurface* aSurface) override;
    void VisitExtendedSurface(KGExtendedSurface<KGMesh>* aSurface) override;
    void VisitSpace(KGSpace* aSpace) override;
    void VisitExtendedSpace(KGExtendedSpace<KGMesh>* aSpace) override;

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

    katrin::KThreeVector fCurrentOrigin;
    katrin::KThreeVector fCurrentXAxis;
    katrin::KThreeVector fCurrentYAxis;
    katrin::KThreeVector fCurrentZAxis;

    KGMeshElementVector* fCurrentElements;

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
