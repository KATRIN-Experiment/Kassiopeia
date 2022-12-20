#ifndef KFMVTKElectrostaticTreeViewer_DEF
#define KFMVTKElectrostaticTreeViewer_DEF

#include "KFMElectrostaticLocalCoefficientFieldCalculator.hh"
#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KThreeVector_KEMField.hh"

#include <limits>

#include <vtkDoubleArray.h>
#include <vtkPolyData.h>
#include <vtkQuad.h>
#include <vtkShortArray.h>
#include <vtkTriangle.h>
#include <vtkUnstructuredGrid.h>

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

    void GenerateGeometryFile(const std::string& fileName = "FastMultipoleTreeData.vtp");
    void GenerateGridFile(const std::string& fileName = "FastMultipoleTreeData.vtu");

    void ViewGeometry();

  private:
    unsigned int fPointCount;

    vtkSmartPointer<vtkPolyData> fPolyData;
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
