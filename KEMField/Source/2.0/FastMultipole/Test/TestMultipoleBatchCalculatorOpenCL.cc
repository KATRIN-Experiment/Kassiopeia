#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KGBox.hh"
#include "KGRectangle.hh"

#include "KGMesher.hh"

#include "KGBEM.hh"
#include "KGBEMConverter.hh"

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
#include "KBiconjugateGradientStabilized.hh"

#include "KIterativeStateWriter.hh"
#include "KIterationTracker.hh"

#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"

#include "KEMConstants.hh"


#include "KFMElectrostaticSurfaceConverter.hh"
#include "KFMElectrostaticElement.hh"
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticMultipoleBatchCalculator_OpenCL.hh"

#include <iostream>
#include <iomanip>

using namespace KGeoBag;
using namespace KEMField;


int main(int /*argc*/, char** /*argv*/)
{
    // Construct the shape
    KGBox* box = new KGBox();
    int meshCount = 2;

    box->SetX0(-.5);
    box->SetX1(.5);
    box->SetXMeshCount(meshCount+1);
    box->SetXMeshPower(3);

    box->SetY0(-.5);
    box->SetY1(.5);
    box->SetYMeshCount(meshCount+2);
    box->SetYMeshPower(3);

    box->SetZ0(-.5);
    box->SetZ1(.5);
    box->SetZMeshCount(meshCount+3);
    box->SetZMeshPower(3);

    KGSurface* cube = new KGSurface(box);
    cube->SetName("box");
    cube->MakeExtension<KGMesh>();
    cube->MakeExtension<KGElectrostaticDirichlet>();
    cube->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

    // Mesh the elements
    KGMesher* mesher = new KGMesher();
    cube->AcceptNode(mesher);

    KSurfaceContainer surfaceContainer;
    KGBEMMeshConverter geometryConverter(surfaceContainer);
    cube->AcceptNode(&geometryConverter);

     //now we have a surface container with a bunch of electrode discretizations
    //we just want to convert these into point clouds, and then bounding balls
    //extract the information we want
    KFMElectrostaticElementContainer<3,1>* elementContainer;
    elementContainer = new KFMElectrostaticElementContainer<3,1>();
    KFMElectrostaticSurfaceConverter* converter = new KFMElectrostaticSurfaceConverter();
    converter->SetSurfaceContainer(&surfaceContainer);
    converter->SetElectrostaticElementContainer(elementContainer);
    converter->Extract();

    KFMElectrostaticMultipoleBatchCalculator_OpenCL* batchCalc = new KFMElectrostaticMultipoleBatchCalculator_OpenCL();

    long buff_size = 64*1024*1024; //bytes
    batchCalc->SetBufferSizeInBytes(buff_size);

    batchCalc->SetDegree(8);
    batchCalc->SetElectrostaticElementContainer(elementContainer);
    batchCalc->Initialize();

    batchCalc->ComputeMoments();

    std::cout<<"OpenCL flags = "<<batchCalc->GetOpenCLFlags()<<std::endl;

    return 0;
}
