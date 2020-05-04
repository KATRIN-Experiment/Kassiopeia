// TestIntegratorSpeedWithKbdInput
// This program tests speed of numeric integration routines from a given Kbd file.
// Author: Daniel Hilk
// Date: 01.05.2016

#include "KBinaryDataStreamer.hh"
#include "KEMConstants.hh"
#include "KEMFileInterface.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KSADataStreamer.hh"
#include "KSerializer.hh"
#include "KSurfaceContainer.hh"
#include "KTypelist.hh"

#include <cstdlib>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KOpenCLSurfaceContainer.hh"
#endif

#include "KElectrostaticIntegratingFieldSolver.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#endif


using namespace KEMField;

double IJKLRANDOM;
void subrn(double* u, int len);
double randomnumber();

clock_t start;

void StartTimer()
{
    start = clock();
}

double Time()
{
    double end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;  // time in seconds
}

int main(int argc, char* argv[])
{
    std::string usage = "\n"
                        "Usage: TestIntegratorSpeedWithKbdInput <options>\n"
                        "\n"
                        "This program tests the speed of numeric integration routines from a given Kbd file..\n"
                        "\n"
                        "\tAvailable options:\n"
                        "\t -h, --help               (shows this message and exits)\n"
                        "\t -f, --file               (specify the input kbd file)\n"
                        "\t -n, --name               (name of the surface container inside kbd file)\n"
                        "\n";

    static struct option longOptions[] = {{"help", no_argument, 0, 'h'},
                                          {"file", required_argument, 0, 'f'},
                                          {"name", required_argument, 0, 'n'}};

    static const char* optString = "h:f:n:";

    std::string inFile = "";
    std::string containerName = "surfaceContainer";

    while (1) {
        char optId = getopt_long(argc, argv, optString, longOptions, NULL);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                break;
            case ('f'):
                inFile = std::string(optarg);
                break;
            case ('n'):
                containerName = std::string(optarg);
                break;
            default:  // unrecognized option
                std::cout << usage << std::endl;
                return 1;
        }
    }

    std::string suffix = inFile.substr(inFile.find_last_of("."), std::string::npos);

    struct stat fileInfo;
    bool exists;
    int fileStat;

    // Attempt to get the file attributes
    fileStat = stat(inFile.c_str(), &fileInfo);
    if (fileStat == 0)
        exists = true;
    else
        exists = false;

    if (!exists) {
        std::cout << "Error: file \"" << inFile << "\" cannot be read." << std::endl;
        return 1;
    }

    KBinaryDataStreamer binaryDataStreamer;

    if (suffix.compare(binaryDataStreamer.GetFileSuffix()) != 0) {
        std::cout << "Error: unkown file extension \"" << suffix << "\"" << std::endl;
        return 1;
    }

    //inspect the files
    KEMFileInterface::GetInstance()->Inspect(inFile);

    // now read in the surface containers
    KSurfaceContainer* surfaceContainer = new KSurfaceContainer();
    KEMFileInterface::GetInstance()->Read(inFile, *surfaceContainer, containerName);

    std::cout << "Surface container with name " << containerName << " in file has size: " << surfaceContainer->size()
              << std::endl;

    // dice field points randomly in cylinder volume
    const double cylZmin(-4.5);
    const double cylZmax(4.5);
    const double cylR(3.5);
    const unsigned int noPoints(100);

    double* fieldPoints = new double[3 * noPoints];

    for (unsigned int i = 0; i < noPoints; i++) {
        IJKLRANDOM = i + 1;

        const double z = randomnumber();
        const double phi = 2. * M_PI * randomnumber();
        const double r = cylR * sqrt(randomnumber());

        fieldPoints[(i * 3)] = cos(phi) * r;                           // x
        fieldPoints[(i * 3) + 1] = sin(phi) * r;                       // y
        fieldPoints[(i * 3) + 2] = cylZmin + z * (cylZmax - cylZmin);  // z

        //KEMField::cout << KThreeVector(fieldPoints[(i*3)],fieldPoints[(i*3)+1],fieldPoints[(i*3)+2]) << KEMField::endl;
    }

#ifdef KEMFIELD_USE_OPENCL
    // surface container
    KOpenCLSurfaceContainer* oclContainer = new KOpenCLSurfaceContainer(*surfaceContainer);
    KOpenCLInterface::GetInstance()->SetActiveData(oclContainer);

    // integrator classes
    KOpenCLElectrostaticBoundaryIntegrator intOCL{KoclEBIFactory::MakeAnalytic(*oclContainer)};
    //KOpenCLElectrostaticBoundaryIntegrator intOCL {
    //    KoclEBIFactory::MakeNumeric( *oclContainer )};
    //KOpenCLElectrostaticBoundaryIntegrator intOCL {
    //    KoclEBIFactory::MakeRWG( *oclContainer )};

    // integrating field solver
    KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>* fOpenCLIntegratingFieldSolver =
        new KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>(*oclContainer, intOCL);

    fOpenCLIntegratingFieldSolver->Initialize();
#else
    KElectrostaticBoundaryIntegrator intNum{KEBIFactory::MakeRWG()};
    KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* fIntegratingFieldSolver =
        new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(*surfaceContainer, intNum);
#endif
    double totalTime = 0.;
    for (unsigned int i = 0; i < noPoints; i++) {
        StartTimer();
#ifdef KEMFIELD_USE_OPENCL
        fOpenCLIntegratingFieldSolver->ElectricFieldAndPotential(
            KThreeVector(fieldPoints[(i * 3)], fieldPoints[(i * 3) + 1], fieldPoints[(i * 3) + 2]));
        //fOpenCLIntegratingFieldSolver->Potential( KThreeVector(fieldPoints[(i*3)],fieldPoints[(i*3)+1],fieldPoints[(i*3)+2]) );
#else
        fIntegratingFieldSolver->ElectricFieldAndPotentialNoKahanSum(
            KThreeVector(fieldPoints[(i * 3)], fieldPoints[(i * 3) + 1], fieldPoints[(i * 3) + 2]));
        totalTime += Time();
#endif
    }
    KEMField::cout << "The computation of electric field and potential simultaneously on " << noPoints
                   << " points took " << totalTime << " seconds." << KEMField::endl;

    delete[] fieldPoints;

    return 0;
}

void subrn(double* u, int len)
{
    // This subroutine computes random numbers u[1],...,u[len]
    // in the (0,1) interval. It uses the 0<IJKLRANDOM<900000000
    // integer as initialization seed.
    //  In the calling program the dimension
    // of the u[] vector should be larger than len (the u[0] value is
    // not used).
    // For each IJKLRANDOM
    // numbers the program computes completely independent random number
    // sequences (see: F. James, Comp. Phys. Comm. 60 (1990) 329, sec. 3.3).

    static int iff = 0;
    static long ijkl, ij, kl, i, j, k, l, ii, jj, m, i97, j97, ivec;
    static float s, t, uu[98], c, cd, cm, uni;
    if (iff == 0) {
        if (IJKLRANDOM == 0) {
            std::cout << "Message from subroutine subrn:\n";
            std::cout << "the global integer IJKLRANDOM should be larger than 0 !!!\n";
            std::cout << "Computation is  stopped !!! \n";
            exit(0);
        }
        ijkl = IJKLRANDOM;
        if (ijkl < 1 || ijkl >= 900000000)
            ijkl = 1;
        ij = ijkl / 30082;
        kl = ijkl - 30082 * ij;
        i = ((ij / 177) % 177) + 2;
        j = (ij % 177) + 2;
        k = ((kl / 169) % 178) + 1;
        l = kl % 169;
        for (ii = 1; ii <= 97; ii++) {
            s = 0;
            t = 0.5;
            for (jj = 1; jj <= 24; jj++) {
                m = (((i * j) % 179) * k) % 179;
                i = j;
                j = k;
                k = m;
                l = (53 * l + 1) % 169;
                if ((l * m) % 64 >= 32)
                    s = s + t;
                t = 0.5 * t;
            }
            uu[ii] = s;
        }
        c = 362436. / 16777216.;
        cd = 7654321. / 16777216.;
        cm = 16777213. / 16777216.;
        i97 = 97;
        j97 = 33;
        iff = 1;
    }
    for (ivec = 1; ivec <= len; ivec++) {
        uni = uu[i97] - uu[j97];
        if (uni < 0.)
            uni = uni + 1.;
        uu[i97] = uni;
        i97 = i97 - 1;
        if (i97 == 0)
            i97 = 97;
        j97 = j97 - 1;
        if (j97 == 0)
            j97 = 97;
        c = c - cd;
        if (c < 0.)
            c = c + cm;
        uni = uni - c;
        if (uni < 0.)
            uni = uni + 1.;
        if (uni == 0.) {
            uni = uu[j97] * 0.59604644775391e-07;
            if (uni == 0.)
                uni = 0.35527136788005e-14;
        }
        u[ivec] = uni;
    }
    return;
}

////////////////////////////////////////////////////////////////

double randomnumber()
{
    // This function computes 1 random number in the (0,1) interval,
    // using the subrn subroutine.

    double u[2];
    subrn(u, 1);
    return u[1];
}
