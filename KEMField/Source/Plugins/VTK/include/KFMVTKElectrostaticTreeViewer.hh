#ifndef KFMVTKElectrostaticTreeViewer_DEF
#define KFMVTKElectrostaticTreeViewer_DEF

#include "KFMElectrostaticLocalCoefficientFieldCalculator.hh"
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KThreeVector_KEMField.hh"

#include <limits>
#include <vtkActor.h>
#include <vtkAppendPolyData.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCleanPolyData.h>
#include <vtkDataSetMapper.h>
#include <vtkDiskSource.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkLine.h>
#include <vtkLinearExtrusionFilter.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkProperty.h>
#include <vtkQuad.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkShortArray.h>
#include <vtkSmartPointer.h>
#include <vtkStripper.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkVector.h>
#include <vtkVoxel.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLPPolyDataWriter.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>

namespace KEMField
{

/**
   * @class KFMVTKElectrostaticTreeViewer
   *
   * @brief A class for rendering fast multipole tree data with VTK.
   *
   */

class KFMVTKElectrostaticTreeViewer : public KFMNodeActor<KFMElectrostaticNode>
{
  public:
    KFMVTKElectrostaticTreeViewer(KFMElectrostaticTree& aTree);
    ~KFMVTKElectrostaticTreeViewer(){};

    void ApplyAction(KFMElectrostaticNode* node);

    void GenerateGeometryFile(std::string fileName = "FastMultipoleTreeData.vtp");
    void GenerateGridFile(std::string fileName = "FastMultipoleTreeData.vtu");

    void ViewGeometry();

  private:
    unsigned int fPointCount;
    vtkSmartPointer<vtkPoints> fPoints;
    vtkSmartPointer<vtkCellArray> fCells;

    vtkSmartPointer<vtkDoubleArray> fTreeLevel;
    vtkSmartPointer<vtkDoubleArray> fOwnedElements;
    vtkSmartPointer<vtkDoubleArray> fDirectCalls;
    vtkSmartPointer<vtkDoubleArray> fZeroOrderMultipole;
    vtkSmartPointer<vtkDoubleArray> fZeroOrderLocalCoeff;

    //these are fast potential/field only
    //they do not include contributions from direct calls, only local coeff expansion
    vtkSmartPointer<vtkDoubleArray> fPotential;
    vtkSmartPointer<vtkDoubleArray> fElectricFieldMagnitude;
    vtkSmartPointer<vtkDoubleArray> fElectricField;

    KFMElectrostaticLocalCoefficientFieldCalculator fFieldCalc;

    //unstructured grid
    vtkSmartPointer<vtkUnstructuredGrid> fGrid;

    //additional scalars we might want to have

    //length
    //electric field at center (vector value)
    //average size of owned elements
    //average size of external direct calls
    //average distance/size ratio for external direct calls
};

}  // namespace KEMField

#endif /* KFMVTKElectrostaticTreeViewer_DEF */
