#include "KFMVTKElectrostaticTreeViewer.hh"

namespace KEMField
{

KFMVTKElectrostaticTreeViewer::KFMVTKElectrostaticTreeViewer(KFMElectrostaticTree& aTree)
{
    fPointCount = 0;
    fPoints = vtkSmartPointer<vtkPoints>::New();
    fCells = vtkSmartPointer<vtkCellArray>::New();

    //tree level
    fTreeLevel = vtkSmartPointer<vtkDoubleArray>::New();
    fTreeLevel->SetName("TreeLevel");

    //n owned primitive
    fOwnedElements = vtkSmartPointer<vtkDoubleArray>::New();
    fOwnedElements->SetName("OwnedElements");

    //n direct calls
    fDirectCalls = vtkSmartPointer<vtkDoubleArray>::New();
    fDirectCalls->SetName("DirectCalls");

    //zeroth order multipole moment (total charge contained)
    fZeroOrderMultipole = vtkSmartPointer<vtkDoubleArray>::New();
    fZeroOrderMultipole->SetName("ZerothOrderMultipoleMoment");

    //zeroth order local coeff (potential at center not including direct calls)
    fZeroOrderLocalCoeff = vtkSmartPointer<vtkDoubleArray>::New();
    fZeroOrderLocalCoeff->SetName("ZerothOrderLocalCoeff");

    fPotential = vtkSmartPointer<vtkDoubleArray>::New();
    fPotential->SetName("Potential");
    fElectricFieldMagnitude = vtkSmartPointer<vtkDoubleArray>::New();
    fElectricFieldMagnitude->SetName("ElectrcFieldMagnitude");

    fElectricField = vtkSmartPointer<vtkDoubleArray>::New();
    fElectricField->SetName("ElectricField");
    fElectricField->SetNumberOfComponents(3);

    //Add the points and hexahedron to an unstructured grid
    fGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();

    fGrid->GetCellData()->AddArray(fTreeLevel);
    fGrid->GetCellData()->AddArray(fOwnedElements);
    fGrid->GetCellData()->AddArray(fDirectCalls);
    fGrid->GetCellData()->AddArray(fZeroOrderMultipole);
    fGrid->GetCellData()->AddArray(fZeroOrderLocalCoeff);
    fGrid->GetCellData()->AddArray(fPotential);
    fGrid->GetCellData()->AddArray(fElectricFieldMagnitude);
    fGrid->GetCellData()->AddArray(fElectricField);

    //TODO uncomment and implement below
    // //average size of owned elements
    // fAverageOwnedElementSize = vtkSmartPointer<vtkDoubleArray>::New();
    // fAverageOwnedElementSize->SetName("AverageOwnedElementSize");
    //
    // //average distance/size ratio for external direct calls
    // fAverageElementDistanceSizeRatio = vtkSmartPointer<vtkDoubleArray>::New();
    // fAverageElementDistanceSizeRatio->SetName("AverageElementDistanceSizeRatio");
    //
    // //electric field at center (not including direct field calls)
    // fElectricField = vtkSmartPointer< vtkVector3<double> >::New();

    //now recursively visit the tree
    aTree.ApplyRecursiveAction(this);
}

void KFMVTKElectrostaticTreeViewer::ApplyAction(KFMElectrostaticNode* node)
{
    if (node != nullptr && !(node->HasChildren()))  //we only visit leaf nodes
    {
        //get owned element ids
        KFMIdentitySet* id_set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMIdentitySet>::GetNodeObject(node);

        //get number of direct calls
        //loop over the node list and collect the direct call elements from their id set lists
        unsigned int subset_size = 0;
        KFMElectrostaticNode* temp_node = node;
        do {
            if (temp_node != nullptr) {
                KFMIdentitySetList* id_set_list =
                    KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMIdentitySetList>::GetNodeObject(temp_node);
                if (id_set_list != nullptr) {
                    unsigned int n_sets = id_set_list->GetNumberOfSets();
                    for (unsigned int j = 0; j < n_sets; j++) {
                        const std::vector<unsigned int>* set = id_set_list->GetSet(j);
                        subset_size += set->size();
                    }
                }
                temp_node = temp_node->GetParent();
            }
        } while (temp_node != nullptr);

        //get multipole moments if they exist
        KFMElectrostaticMultipoleSet* mult_mom =
            KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet>::GetNodeObject(node);
        double monopole_moment = 0.0;
        if (mult_mom != nullptr) {
            std::vector<std::complex<double>> moments;
            mult_mom->GetMoments(&moments);
            monopole_moment = moments[0].real();
        }

        //get local coefficients if they exist
        KFMElectrostaticLocalCoefficientSet* local_coeff =
            KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);
        double monopole_coeff = 0.0;
        if (local_coeff != nullptr) {
            std::vector<std::complex<double>> coeffs;
            local_coeff->GetMoments(&coeffs);
            monopole_coeff = coeffs[0].real();

            fFieldCalc.SetDegree(local_coeff->GetDegree());
        }

        //retrieve the cube/spatial information
        KFMCube<3>* cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3>>::GetNodeObject(node);

        // fFieldCalc.SetDegree(local_coeff->GetDegree());
        fFieldCalc.SetExpansionOrigin(cube->GetCenter());
        fFieldCalc.SetLocalCoefficients(local_coeff);

        double potential = fFieldCalc.Potential(cube->GetCenter());
        double field[3];
        fFieldCalc.ElectricField(cube->GetCenter(), field);
        double field_mag = std::sqrt(field[0] * field[0] + field[1] * field[1] + field[2] * field[2]);


        //add the cube's vertices
        unsigned cube_start_index = fPointCount;
        for (unsigned int i = 0; i < 8; i++) {
            KFMPoint<3> corner = cube->GetCorner(i);
            fPoints->InsertNextPoint(corner[0], corner[1], corner[2]);
            fPointCount++;
        }

        //face vertex lists
        unsigned int face_ids[6][4] = {
            {0, 1, 3, 2},
            {0, 1, 5, 4},
            {1, 3, 7, 5},
            {5, 7, 6, 4},
            {2, 3, 7, 6},
            {0, 2, 6, 4},
        };

        //create a quad for each face of the node's cube
        for (auto& face_id : face_ids) {
            //face cells from the corner points
            vtkSmartPointer<vtkQuad> face = vtkSmartPointer<vtkQuad>::New();

            for (unsigned int i = 0; i < 4; i++) {
                // KFMPoint<3> corner = cube->GetCorner(face_ids[f][i]);
                // fPoints->InsertNextPoint(corner[0], corner[1], corner[2]);
                face->GetPointIds()->SetId(i, cube_start_index + face_id[i]);
            }
            //now insert the new hexahedron into the cell array
            fCells->InsertNextCell(face);

            //add the data for this cell
            fTreeLevel->InsertNextValue(node->GetLevel());
            if (id_set != nullptr) {
                fOwnedElements->InsertNextValue(id_set->GetSize());
            }
            else {
                fOwnedElements->InsertNextValue(0);
            }

            fDirectCalls->InsertNextValue(subset_size);
            fZeroOrderMultipole->InsertNextValue(monopole_moment);
            fZeroOrderLocalCoeff->InsertNextValue(monopole_coeff);

            fPotential->InsertNextValue(potential);
            fElectricFieldMagnitude->InsertNextValue(field_mag);

            fElectricField->InsertNextTuple(field);

            fGrid->SetPoints(fPoints);
            fGrid->InsertNextCell(face->GetCellType(), face->GetPointIds());
        }
    }
}

void KFMVTKElectrostaticTreeViewer::GenerateGeometryFile(const std::string& fileName)
{
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(fPoints);
    polydata->SetPolys(fCells);
    // polydata->GetPointData()->AddArray(fTreeLevel);
    // polydata->GetPointData()->AddArray(fOwnedElements);
    // polydata->GetPointData()->AddArray(fDirectCalls);
    // polydata->GetPointData()->AddArray(fZeroOrderMultipole);
    // polydata->GetPointData()->AddArray(fZeroOrderLocalCoeff);

    polydata->GetCellData()->AddArray(fTreeLevel);
    polydata->GetCellData()->AddArray(fOwnedElements);
    polydata->GetCellData()->AddArray(fDirectCalls);
    polydata->GetCellData()->AddArray(fZeroOrderMultipole);
    polydata->GetCellData()->AddArray(fZeroOrderLocalCoeff);
    polydata->GetCellData()->AddArray(fPotential);
    polydata->GetCellData()->AddArray(fElectricFieldMagnitude);
    polydata->GetCellData()->AddArray(fElectricField);

    vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();

    // writer->SetDataModeToAscii();
    writer->SetDataModeToBinary();
    writer->SetCompressorTypeToZLib();
    writer->SetIdTypeToInt64();

#if VTK_MAJOR_VERSION <= 5
    writer->SetInput(polydata);
#else
    writer->SetHeaderTypeToUInt64();
    writer->SetInputData(polydata);
#endif
    writer->SetFileName(fileName.c_str());
    writer->Write();
}

void KFMVTKElectrostaticTreeViewer::GenerateGridFile(const std::string& fileName)
{
    // Write file
    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
#if VTK_MAJOR_VERSION <= 5
    writer->SetInput(fGrid);
#else
    writer->SetInputData(fGrid);
#endif
    writer->SetFileName(fileName.c_str());
    writer->Write();
}

void KFMVTKElectrostaticTreeViewer::ViewGeometry()
{
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(fPoints);
    polydata->SetPolys(fCells);
    polydata->GetCellData()->AddArray(fTreeLevel);
    polydata->GetCellData()->AddArray(fOwnedElements);
    polydata->GetCellData()->AddArray(fDirectCalls);
    polydata->GetCellData()->AddArray(fZeroOrderMultipole);
    polydata->GetCellData()->AddArray(fZeroOrderLocalCoeff);
    polydata->GetCellData()->AddArray(fPotential);
    polydata->GetCellData()->AddArray(fElectricFieldMagnitude);
    polydata->GetCellData()->AddArray(fElectricField);

    vtkSmartPointer<vtkTriangleFilter> trifilter = vtkSmartPointer<vtkTriangleFilter>::New();
#ifdef VTK6
    trifilter->SetInputData(polydata);
#else
    trifilter->SetInput(polydata);
#endif

    vtkSmartPointer<vtkStripper> stripper = vtkSmartPointer<vtkStripper>::New();
    stripper->SetInputConnection(trifilter->GetOutputPort());

    // Create an actor and mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#ifdef VTK6
    mapper->SetInputConnection(stripper->GetOutputPort());
#else
    mapper->SetInput(stripper->GetOutput());
#endif

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetRepresentationToWireframe();

    // Create a renderer, render window, and interactor
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(750, 750);
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderer->AddActor(actor);
    renderWindow->Render();
    renderWindowInteractor->Start();
}

}  // namespace KEMField
