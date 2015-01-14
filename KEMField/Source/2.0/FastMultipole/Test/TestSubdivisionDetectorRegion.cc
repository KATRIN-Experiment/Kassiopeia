#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KGDetectorRegion.hh"
#include "KGDeterministicMesher.hh"

#include "KGEMConverter.hh"

#ifdef KEMFIELD_USE_VTK
#include "KEMVTKViewer.hh"
#endif

#include "KTypelist.hh"
#include "KSurfaceTypes.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include "KDataDisplay.hh"

#include "KElectrostaticBoundaryIntegrator.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"

#include "KGaussianElimination.hh"
#include "KRobinHood.hh"

#include "KIterativeStateSaver.hh"
#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"

#include "KEMConstants.hh"



#include "KFMMath.hh"
#include "KFMCube.hh"
#include "KFMBall.hh"
#include "KFMIdentityPair.hh"
#include "KFMObjectContainer.hh"
#include "KFMCubicSpaceTree.hh"
#include "KFMCubicSpaceTreeProperties.hh"
#include "KFMCubicSpaceBallSorter.hh"
#include "KFMInsertionCondition.hh"
#include "KFMSubdivisionCondition.hh"
#include "KFMElectrostaticSurfaceConverter.hh"
#include "KFMPointCloudToBoundingBallConverter.hh"

#include "KFMElectrostaticElement.hh"
#include "KFMElectrostaticElementContainer.hh"


#include <iostream>
#include <iomanip>

using namespace KGeoBag;
using namespace KEMField;

//define the typelist of objects that are attached to a node
typedef KTYPELIST_3(KFMCube<3>, KFMIdentitySet, KFMCubicSpaceTreeProperties<3>) cubic_node_objects;

int main(int /*argc*/, char** /*argv*/)
{

    double scale = 1.5; //this doesnt seem to change anything

    std::cout << std::setprecision(15);

 // Create the detector region
  KGDetectorRegion* detectorRegion = new KGDetectorRegion();
  detectorRegion->SetScale(scale);

  detectorRegion->IsGateValve(true);
  detectorRegion->IsPortValve(true);
  detectorRegion->IsPortValveEntry(true);
  detectorRegion->IsPortValveWall(true);
  detectorRegion->IsThermalBreak(true);
  detectorRegion->IsElectrodeInside(true);
  detectorRegion->IsCoolBand(true);
  detectorRegion->IsInsulatorTaper(true);
  detectorRegion->IsCopperElectrode(true);
  detectorRegion->IsFPDHolder(true);
  detectorRegion->IsWafer(true);
  detectorRegion->IsDetectorBase(true);
  detectorRegion->IsECrossGround(true);
  detectorRegion->IsECrossHighVolt(true);
  detectorRegion->IsQuartz(true);
  detectorRegion->IsECrossGround2(true);
  detectorRegion->IsECrossHighVolt2(true);
  detectorRegion->IsQuartz2(true);
  detectorRegion->IsFinalConflatTorque(true);
  detectorRegion->IsPortValveIntersection(true);
  detectorRegion->IsCalibrationDisk(true);
  detectorRegion->IsVacuumHouseFlangeA(true);
  detectorRegion->Is14Step(true);
  detectorRegion->IsVacuumHousePipe(true);
  detectorRegion->IsDetectorVacuumFlange(true);
  detectorRegion->IsPinchMagnet(true);
  detectorRegion->IsDetectorMagnet(true);


//  detectorRegion->IsGateValve(true);
//  detectorRegion->IsPortValve(false);
//  detectorRegion->IsPortValveEntry(false);
//  detectorRegion->IsPortValveWall(false);
//  detectorRegion->IsThermalBreak(false);
//  detectorRegion->IsElectrodeInside(false);
//  detectorRegion->IsCoolBand(false);
//  detectorRegion->IsInsulatorTaper(false);
//  detectorRegion->IsCopperElectrode(false);
//  detectorRegion->IsFPDHolder(false);
//  detectorRegion->IsWafer(false);
//  detectorRegion->IsDetectorBase(false);
//  detectorRegion->IsECrossGround(false);
//  detectorRegion->IsECrossHighVolt(false);
//  detectorRegion->IsQuartz(false);
//  detectorRegion->IsECrossGround2(false);
//  detectorRegion->IsECrossHighVolt2(false);
//  detectorRegion->IsQuartz2(false);
//  detectorRegion->IsFinalConflatTorque(false);
//  detectorRegion->IsPortValveIntersection(false);
//  detectorRegion->IsCalibrationDisk(false);
//  detectorRegion->IsVacuumHouseFlangeA(false);
//  detectorRegion->Is14Step(false);
//  detectorRegion->IsVacuumHousePipe(false);
//  detectorRegion->IsDetectorVacuumFlange(false);
//  detectorRegion->IsPinchMagnet(false);
//  detectorRegion->IsDetectorMagnet(false);




    detectorRegion->Initialize();

    // Mesh the elements
    KGDeterministicMesher* mesher = new KGDeterministicMesher();
    mesher->RecursiveVisit(detectorRegion);

    KSurfaceContainer surfaceContainer;

    KGEMConverter geometryConverter(surfaceContainer);
    geometryConverter.RecursiveVisit(detectorRegion);

    //create the tree object
    KFMCubicSpaceTree<3, cubic_node_objects>* tree = new KFMCubicSpaceTree<3, cubic_node_objects>();
    tree->GetTreeProperties()->SetTreeID(0);
    tree->GetTreeProperties()->SetMaxTreeDepth(3);
    tree->GetTreeProperties()->SetCubicNeighborOrder(1);
    unsigned int dim[3] = {3,3,3};
    tree->GetTreeProperties()->SetDimensions(dim);

    //conditions for subdivision of a node
    KFMInsertionCondition<3> basic_insertion_condition;
    KFMSubdivisionCondition<3, cubic_node_objects> sub_cond;
    sub_cond.SetInsertionCondition(&basic_insertion_condition);
    tree->SetSubdivisionCondition(&sub_cond);

    //things to do on a node after it has been visited by the progenitor
    KFMCubicSpaceBallSorter<3, cubic_node_objects> bball_sorter;
    bball_sorter.SetInsertionCondition(&basic_insertion_condition);
    tree->AddPostSubdivisionAction(&bball_sorter);

     //now we have a surface container with a bunch of electrode discretizations
    //we just want to convert these into point clouds, and then bounding balls
    //extract the information we want
    KFMElectrostaticElementContainer<3,1> elementContainer;
    KFMElectrostaticSurfaceConverter converter;
    converter.SetSurfaceContainer(&surfaceContainer);
    converter.SetElectrostaticElementContainer(&elementContainer);
    converter.Extract();

    KFMObjectContainer< KFMBall<3> >* bballs = elementContainer.GetBoundingBallContainer();
    unsigned int n_bballs = bballs->GetNObjects();


    sub_cond.SetBoundingBallContainer( bballs );
    bball_sorter.SetBoundingBallContainer( bballs );

    std::cout<<"# vtk DataFile Version 2.0"<<std::endl;
    std::cout<<"Unstructured grid legacy vtk file with point scalar data"<<std::endl;
    std::cout<<"ASCII"<<std::endl;

    std::cout<<"DATASET UNSTRUCTURED_GRID"<<std::endl;
    std::cout<<"POINTS "<<n_bballs<<" double"<<std::endl;

    for(unsigned int i=0; i<n_bballs; i++)
    {
        KFMPoint<3> center;
        center = bballs->GetObjectWithID(i)->GetCenter();
        std::cout<<center[0]<<" "<<center[1]<<" "<<center[2]<<std::endl;
    }

    std::cout<<std::endl;
    std::cout<<"POINT_DATA  "<<n_bballs<<std::endl;
    std::cout<<"SCALARS diameter double"<<std::endl;
    std::cout<<"LOOKUP_TABLE default"<<std::endl;

    for(unsigned int i=0; i<n_bballs; i++)
    {
        std::cout<<2.0*(bballs->GetObjectWithID(i)->GetRadius())<<std::endl;
    }


    KFMCube<3> root_cube;
    double origin[3] = {0.,0.,0.};
    double base_length = 2.5;
    root_cube.SetParameters(origin, base_length);

    KFMNode<cubic_node_objects>* root_node = tree->GetRootNode();

    //attach world cube to root node
    KFMObjectRetriever<cubic_node_objects, KFMCube<3> >::SetNodeObject(&root_cube, root_node);

    //attach complete bounding ball list to root node
    KFMIdentitySet root_list;
    for(unsigned int i=0; i<n_bballs; i++)
    {
        root_list.AddID(i);
    }

    KFMObjectRetriever<cubic_node_objects, KFMIdentitySet >::SetNodeObject(&root_list, root_node);

    //finally attach tree properties
    KFMObjectRetriever<cubic_node_objects, KFMCubicSpaceTreeProperties<3> >::SetNodeObject(tree->GetTreeProperties(), root_node);

    tree->ConstructTree();

    return 0;


}
