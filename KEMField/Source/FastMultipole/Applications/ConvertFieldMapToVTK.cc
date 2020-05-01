#include "KEMFileInterface.hh"
#include "KFMNamedScalarData.hh"
#include "KFMNamedScalarDataCollection.hh"

#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef KEMFIELD_USE_VTK
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
#include <vtkPolygon.h>
#include <vtkProperty.h>
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
#include <vtkVersion.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLPolyDataWriter.h>
#endif /* KEMFIELD_USE_VTK */


using namespace KEMField;

int main(int argc, char** argv)
{


    if (argc != 2) {
        std::cout << "please give path to file" << std::endl;
        return 1;
    }

    std::string filename(argv[1]);

    KSAObjectInputNode<KFMNamedScalarDataCollection>* data_node =
        new KSAObjectInputNode<KFMNamedScalarDataCollection>("data_collection");

    bool result;
    KEMFileInterface::GetInstance()->ReadKSAFile(data_node, filename, result);

    if (!result) {
        std::cout << "failed to read file" << std::endl;
        return 1;
    };

    KFMNamedScalarDataCollection* data = data_node->GetObject();

    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////

    //Create a vtkPoints object and store the points in it
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    KFMNamedScalarData* x_coord = data->GetDataWithName(std::string("x_coordinate"));
    KFMNamedScalarData* y_coord = data->GetDataWithName(std::string("y_coordinate"));
    KFMNamedScalarData* z_coord = data->GetDataWithName(std::string("z_coordinate"));

    unsigned int n_points = x_coord->GetSize();
    double p[3];
    for (unsigned int i = 0; i < n_points; i++) {
        p[0] = x_coord->GetValue(i);
        p[1] = y_coord->GetValue(i);
        p[2] = z_coord->GetValue(i);
        points->InsertNextPoint(p);
    }

    // Create a polydata to store everything in
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    // Add the points to the dataset
    polyData->SetPoints(points);

    //collect the rest of the (non-point) scalar data
    unsigned int n_data_sets = data->GetNDataSets();
    for (unsigned int i = 0; i < n_data_sets; i++) {
        std::string name = data->GetDataSetWithIndex(i)->GetName();
        if ((name != (std::string("x_coordinate"))) && (name != (std::string("y_coordinate"))) &&
            (name != (std::string("z_coordinate")))) {
            if ((name != (std::string("fmm_time_per_potential_call"))) &&
                (name != (std::string("fmm_time_per_field_call"))) &&
                (name != (std::string("direct_time_per_potential_call"))) &&
                (name != (std::string("direct_time_per_field_call")))) {

                vtkSmartPointer<vtkDoubleArray> array;
                array = vtkSmartPointer<vtkDoubleArray>::New();
                array->SetName((data->GetDataSetWithIndex(i)->GetName()).c_str());
                array->Initialize();

                unsigned int size = data->GetDataSetWithIndex(i)->GetSize();
                for (unsigned int j = 0; j < size; j++) {
                    array->InsertNextValue(data->GetDataSetWithIndex(i)->GetValue(j));
                }

                polyData->GetPointData()->AddArray(array);
            }
        }
    }


    vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName("./test_vtk.vtk");

#if VTK_MAJOR_VERSION > 5
    writer->SetInputData(polyData);
#else
    writer->SetInput(polyData);
#endif

    writer->Write();


    return 0;
}
