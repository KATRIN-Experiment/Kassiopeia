#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <sys/stat.h>

#include "KTypelist.hh"
#include "KSurfaceTypes.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include "KDataDisplay.hh"


#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"
#include "KSerializer.hh"
#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"

#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"

#include "KGaussianElimination.hh"
#include "KRobinHood.hh"

#include "KEMFieldCanvas.hh"

#ifdef KEMFIELD_USE_VTK
#include "KEMVTKViewer.hh"
#include "KVTKIterationPlotter.hh"
#include "KEMVTKFieldCanvas.hh"
#endif

#include "KEMConstants.hh"

#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticFieldMapper_SingleThread.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeConstructor.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"

#include "KFMNamedScalarData.hh"
#include "KFMNamedScalarDataCollection.hh"


#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh"
#include "KFMElectrostaticFieldMapper_OpenCL.hh"
#endif




#include <getopt.h>
#include <iostream>
#include <sys/stat.h>

#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"

#include "KFMElectrostaticBasisDataExtractor.hh"
#include "KSurfaceContainer.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#endif

#ifdef KEMFIELD_USE_GSL
#include <gsl/gsl_rng.h>
#endif

using namespace KEMField;

int main(int argc, char* argv[])
{

    std::string usage =
    "\n"
    "Usage: CompareChargeDensities <options>\n"
    "\n"
    "This program takes two KEMField files and compares the charge density values. These files must contain the same geometry.\n"
    "\n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\t -a, --fileA              (specify the first file)\n"
    "\t -b, --fileB              (specify the second file)\n"
    "\t -n, --nameA              (specify the surface container name in file A)\n"
    "\t -m, --nameB              (specify the surface container name in file B)\n"
    "\t -s, --size               (size of box for potential comparison)\n"
    "\n";

    static struct option longOptions[] = {
        {"help", no_argument, 0, 'h'},
        {"fileA", required_argument, 0, 'a'},
        {"fileB", required_argument, 0, 'b'},
        {"nameA", required_argument, 0, 'n'},
        {"nameB", required_argument, 0, 'm'},
        {"size", required_argument, 0, 's'}
    };

    static const char *optString = "ha:b:n:m:s:";

    std::string inFile1 = "";
    std::string inFile2 = "";
    std::string containerName1 = "surfaceContainer";
    std::string containerName2 = "surfaceContainer";
    double len = 1.0;

    while(1)
    {
        char optId = getopt_long(argc, argv,optString, longOptions, NULL);
        if(optId == -1) break;
        switch(optId) {
        case('h'): // help
            std::cout<<usage<<std::endl;
        break;
        case('a'):
            inFile1 = std::string(optarg);
        break;
        case('b'):
            inFile2 = std::string(optarg);
        break;
        case ('n'):
            containerName1 = std::string(optarg);
        break;
        case ('m'):
            containerName2 = std::string(optarg);
        break;
        case ('s'):
            len = atof(optarg);
        break;
        default: // unrecognized option
            std::cout<<usage<<std::endl;
        return 1;
        }
    }

    std::string suffix1 = inFile1.substr(inFile1.find_last_of("."),std::string::npos);
    std::string suffix2 = inFile2.substr(inFile2.find_last_of("."),std::string::npos);

    struct stat fileInfo1;
    bool exists1;
    int fileStat1;

    // Attempt to get the file attributes
    fileStat1 = stat(inFile1.c_str(),&fileInfo1);
    if(fileStat1 == 0)
    exists1 = true;
    else
    exists1 = false;

    if (!exists1)
    {
    std::cout<<"Error: file \""<<inFile1<<"\" cannot be read."<<std::endl;
    return 1;
    }

    struct stat fileInfo2;
    bool exists2;
    int fileStat2;

    // Attempt to get the file attributes
    fileStat2 = stat(inFile2.c_str(),&fileInfo2);
    if(fileStat2 == 0)
    exists2 = true;
    else
    exists2 = false;

    if (!exists2)
    {
        std::cout<<"Error: file \""<<inFile2<<"\" cannot be read."<<std::endl;
        return 1;
    }

    KBinaryDataStreamer binaryDataStreamer;

    if (suffix1.compare(binaryDataStreamer.GetFileSuffix()) != 0)
    {
        std::cout<<"Error: unkown file extension \""<<suffix1<<"\""<<std::endl;
        return 1;
    }

    if (suffix2.compare(binaryDataStreamer.GetFileSuffix()) != 0)
    {
        std::cout<<"Error: unkown file extension \""<<suffix2<<"\""<<std::endl;
        return 1;
    }

    //inspect the files
    KEMFileInterface::GetInstance()->Inspect(inFile1);
    KEMFileInterface::GetInstance()->Inspect(inFile2);

    //now read in the surface containers
    KSurfaceContainer surfaceContainer1;
    KSurfaceContainer surfaceContainer2;
    KEMFileInterface::GetInstance()->Read(inFile1,surfaceContainer1,containerName1);
    KEMFileInterface::GetInstance()->Read(inFile2,surfaceContainer2,containerName2);

    std::cout<<"Surface container with name "<<containerName1<<" in file 1 has size: "<<surfaceContainer1.size()<<std::endl;
    std::cout<<"Surface container with name "<<containerName2<<" in file 2 has size: "<<surfaceContainer2.size()<<std::endl;

    //hash the surface container elements to make sure they share the same geometry
    int HashMaskedBits = 20;
    double HashThreshold = 1.e-14;

    // compute hash of the bare geometry
    KMD5HashGenerator tShapeHashGenerator1;
    tShapeHashGenerator1.MaskedBits( HashMaskedBits );
    tShapeHashGenerator1.Threshold( HashThreshold );
    tShapeHashGenerator1.Omit( Type2Type< KElectrostaticBasis >() );
    tShapeHashGenerator1.Omit( Type2Type< KBoundaryType< KElectrostaticBasis, KDirichletBoundary > >() );
    tShapeHashGenerator1.Omit( Type2Type< KBoundaryType< KElectrostaticBasis, KNeumannBoundary > >() );
    std::string fHash1 = tShapeHashGenerator1.GenerateHash( surfaceContainer1 );

    KMD5HashGenerator tShapeHashGenerator2;
    tShapeHashGenerator2.MaskedBits( HashMaskedBits );
    tShapeHashGenerator2.Threshold( HashThreshold );
    tShapeHashGenerator2.Omit( Type2Type< KElectrostaticBasis >() );
    tShapeHashGenerator2.Omit( Type2Type< KBoundaryType< KElectrostaticBasis, KDirichletBoundary > >() );
    tShapeHashGenerator2.Omit( Type2Type< KBoundaryType< KElectrostaticBasis, KNeumannBoundary > >() );
    std::string fHash2 = tShapeHashGenerator2.GenerateHash( surfaceContainer2 );

    std::cout<<"Hash of container 1: "<<fHash1<<std::endl;
    std::cout<<"Hash of container 2: "<<fHash2<<std::endl;

    if(fHash1 != fHash2)
    {
        //error, surface containers do not have the same geometry
        std::cout<<"Warning the geometry hash of the surface containers does not match."<<std::endl;
        std::cout<<"It is possible these two files contain different geometries."<<std::endl;
    }

    if(surfaceContainer1.size() != surfaceContainer2.size())
    {
        //error, surface containers do not have the same geometry
        std::cout<<"Cannot compare geometries with different sizes "<<std::endl;
        std::cout<<"size of container 1: "<<surfaceContainer1.size()<<std::endl;
        std::cout<<"size of container 2: "<<surfaceContainer2.size()<<std::endl;
        return 1;
    }
    else
    {
        KFMElectrostaticBasisDataExtractor basisExtractor; //only operates on triangles/rectangles/wires

        unsigned int size = surfaceContainer1.size();
        //loop over every element in the container and retrieve the charge density
        //compute the difference and collect the global absolute L2 and L_inf errors
        double L2_diff = 0.0;
        double L2_norm1 = 0.0;
        double L2_norm2 = 0.0;
        double Linf_diff = 0.0;
        for(unsigned int i=0; i < size; i++)
        {
            double cd1 = 0;
            double cd2 = 0;

            //extract the basis data
            double area1 = 0.0;
            surfaceContainer1.at(i)->Accept(basisExtractor);
            area1 = surfaceContainer1.at(i)->GetShape()->Area();
            KFMBasisData<1> basis1 = basisExtractor.GetBasisData();
            cd1 = area1*basis1[0];

            //extract the basis data
            double area2 = 0.0;
            surfaceContainer2.at(i)->Accept(basisExtractor);
            area2 = surfaceContainer2.at(i)->GetShape()->Area();
            KFMBasisData<1> basis2 = basisExtractor.GetBasisData();
            cd2 = area2*basis2[0];

            double diff = cd1 - cd2;
            L2_diff += diff*diff;
            L2_norm1 += cd1*cd1;
            L2_norm2 += cd2*cd2;

            if(Linf_diff < std::fabs(diff) ){Linf_diff = std::fabs(diff);};
        }

        L2_diff = std::sqrt(L2_diff);
        L2_norm1 = std::sqrt(L2_norm1);
        L2_norm2 = std::sqrt(L2_norm2);

        std::cout<<"Absolute L2 norm difference = "<<L2_diff<<std::endl;

        std::cout<<"L2 norm of charge density data in "<<inFile1<<": = "<<L2_norm1<<std::endl;
        std::cout<<"L2 norm of charge density data in "<<inFile2<<": = "<<L2_norm2<<std::endl;

        std::cout<<"Relative L2 norm difference w.r.t. "<<inFile1<<":  = "<<L2_diff/L2_norm1<<std::endl;
        std::cout<<"Relative L2 norm difference w.r.t. "<<inFile2<<":  = "<<L2_diff/L2_norm2<<std::endl;

        std::cout<<"Absolute L_inf difference = "<<Linf_diff<<std::endl;



        //now create the direct solver
        #ifdef KEMFIELD_USE_OPENCL
        KOpenCLSurfaceContainer* oclContainer1;
        oclContainer1 = new KOpenCLSurfaceContainer( surfaceContainer1 );
        KOpenCLInterface::GetInstance()->SetActiveData( oclContainer1 );
        KOpenCLElectrostaticBoundaryIntegrator integrator1{KoclEBIFactory::MakeDefault(*oclContainer1)};
        KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>* direct_solver1 = new KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>(*oclContainer1,integrator1);
        direct_solver1->Initialize();
        #else
        KElectrostaticBoundaryIntegrator integrator1 {KEBIFactory::MakeDefault()};
        KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* direct_solver1 = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer1,integrator1);
        #endif


        //now create the direct solver
        #ifdef KEMFIELD_USE_OPENCL
        KOpenCLSurfaceContainer* oclContainer2;
        oclContainer2 = new KOpenCLSurfaceContainer( surfaceContainer2 );
        KOpenCLInterface::GetInstance()->SetActiveData( oclContainer2 );
        KOpenCLElectrostaticBoundaryIntegrator integrator2{KoclEBIFactory::MakeDefault(*oclContainer2)};
        KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>* direct_solver2 = new KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>(*oclContainer2,integrator2);
        direct_solver2->Initialize();
        #else
        KElectrostaticBoundaryIntegrator integrator2 {KEBIFactory::MakeDefault()};
        KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* direct_solver2 = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer2,integrator2);
        #endif

        #ifdef KEMFIELD_USE_GSL
        const gsl_rng_type* T;
        gsl_rng_env_setup();
        T = gsl_rng_default; //default is mt199937
        gsl_rng* fR = gsl_rng_alloc(T);
        #endif




        std::vector< KEMThreeVector > random_points;

        for(unsigned int i=0; i<100; i++)
        {
            double x = 0;
            double y = 0;
            double z = 0;

            #ifndef KEMFIELD_USE_GSL
                //we don't need high quality random numbers here, so we use rand()
                double m = RAND_MAX;
                m += 1;// do not want the range to be inclusive of the upper limit
                double r1 = rand();
                x = len*(r1/m);
                r1 = rand();
                y = len*( r1/m);
                r1 = rand();
                z = len*(r1/m);
            #else
                //gsl is available, so use it instead
                x = len*(gsl_rng_uniform(fR));
                y = len*(gsl_rng_uniform(fR));
                z = len*(gsl_rng_uniform(fR));
            #endif

            random_points.push_back( KEMThreeVector(x,y,z) );
        }

        double l2_pot_diff = 0.0;

        for(unsigned int i=0; i<100; i++)
        {
            double pot1 = direct_solver1->Potential(random_points[i]);
            double pot2 = direct_solver2->Potential(random_points[i]);

            std::cout<<"pot1 = "<<pot1<<std::endl;
            std::cout<<"pot2 = "<<pot2<<std::endl;

            double delta = pot1 - pot2;
            l2_pot_diff += delta*delta;
        }

        std::cout<<"absolute l2 pot diff = "<<std::sqrt(l2_pot_diff)<<std::endl;

    }

  return 0;
}
