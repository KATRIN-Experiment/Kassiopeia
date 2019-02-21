#ifndef KFMVTKElectrostaticTreeViewer_DEF
#define KFMVTKElectrostaticTreeViewer_DEF

#include <limits>

#include <vtkTriangle.h>
#include <vtkQuad.h>
#include <vtkVoxel.h>
#include <vtkPoints.h>
#include <vtkVector.h>
#include <vtkDataSetMapper.h>
#include <vtkProperty.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDiskSource.h>
#include <vtkLinearExtrusionFilter.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkAppendPolyData.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkShortArray.h>
#include <vtkDoubleArray.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkPolyData.h>
#include <vtkDataSetMapper.h>
#include <vtkLine.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkTriangleFilter.h>
#include <vtkStripper.h>
#include <vtkCleanPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLPPolyDataWriter.h>
#include <vtkActor.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include "KThreeVector_KEMField.hh"

#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticLocalCoefficientFieldCalculator.hh"

namespace KEMField
{

  /**
   * @class KFMVTKElectrostaticTreeViewer
   *
   * @brief A class for rendering fast multipole tree data with VTK.
   *
   */

    class KFMVTKElectrostaticTreeViewer : public KFMNodeActor< KFMElectrostaticNode >
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

}

#endif /* KFMVTKElectrostaticTreeViewer_DEF */
