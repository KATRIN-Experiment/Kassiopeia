#ifndef KGVTKNEARESTNORMALPAINTER_HH_
#define KGVTKNEARESTNORMALPAINTER_HH_

#include "KField.h"
#include "KGBall.hh"
#include "KGCore.hh"
#include "KGCube.hh"
#include "KGMesh.hh"
#include "KGMeshElementCollector.hh"
#include "KGNavigableMeshElementContainer.hh"
#include "KGNavigableMeshFirstIntersectionFinder.hh"
#include "KGNavigableMeshIntersectionFinder.hh"
#include "KGNavigableMeshTree.hh"
#include "KGNavigableMeshTreeBuilder.hh"
#include "KGNavigableMeshTreeInformationExtractor.hh"
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

class KGVTKMeshIntersectionTester : public katrin::KVTKPainter
{
  public:
    KGVTKMeshIntersectionTester();
    ~KGVTKMeshIntersectionTester() override;

    void Render() override;
    void Display() override;
    void Write() override;

    void AddSurface(KGSurface* aSurface);
    void AddSpace(KGSpace* aSpace);

    void SetSampleCount(unsigned int s)
    {
        fSampleCount = s;
    };
    void SetSampleColor(const KGRGBColor& c)
    {
        fSampleColor = c;
    };
    void SetPointColor(const KGRGBColor& c)
    {
        fPointColor = c;
    };
    void SetUnintersectedLineColor(const KGRGBColor& c)
    {
        fUnintersectedLineColor = c;
    };
    void SetIntersectedLineColor(const KGRGBColor& c)
    {
        fIntersectedLineColor = c;
    };
    void SetVertexSize(double s)
    {
        fVertexSize = s;
    };
    void SetLineSize(double s)
    {
        fLineSize = s;
    };

  private:
    void Construct();

    //mesh container and tree
    KGNavigableMeshElementContainer fContainer;
    KGNavigableMeshTree fTree;
    //KGNavigableMeshFirstIntersectionFinder fFirstIntersectionCalculator;
    KGNavigableMeshFirstIntersectionFinder fIntersectionCalculator;
    KGBall<KGMESH_DIM> fBoundingBall;


    unsigned int fSampleCount;
    KGRGBColor fSampleColor;
    KGRGBColor fPointColor;
    KGRGBColor fUnintersectedLineColor;
    KGRGBColor fIntersectedLineColor;
    double fVertexSize;
    double fLineSize;

    //surface arrays
    std::vector<KGSurface*> fSurfaces;
    std::vector<KGSpace*> fSpaces;

    //vtk data
    vtkSmartPointer<vtkPoints> fPoints;
    vtkSmartPointer<vtkUnsignedCharArray> fColors;
    vtkSmartPointer<vtkCellArray> fPointCells;
    vtkSmartPointer<vtkCellArray> fLineCells;
    vtkSmartPointer<vtkPolyData> fPolyData;
    vtkSmartPointer<vtkPolyDataMapper> fMapper;
    vtkSmartPointer<vtkActor> fActor;
};

inline void KGVTKMeshIntersectionTester::AddSurface(KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}

inline void KGVTKMeshIntersectionTester::AddSpace(KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}


}  // namespace KGeoBag

#endif
