#include "KMessage.h"
#include "KSFieldFinder.h"
#include "KSMainMessage.h"
#include "KSRootMagneticField.h"
#include "KSTrajTrajectoryMagnetic.h"
#include "KSTrajIntegratorRK8.h"
#include "KSTrajInterpolatorFast.h"
#include "KSTrajTermPropagation.h"
#include "KSTrajControlTime.h"
#include "KTextFile.h"
#include "KThreeVector.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"


using namespace Kassiopeia;
using namespace katrin;
using namespace KGeoBag;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 9) {
        cout
            << "usage: ./SimpleMagneticFieldCalculatorAlongFieldline <config_file.xml> <x> <y> <z> <dist> <nsteps> <output_file.txt> <magnetic_field_name1> [<magnetic_field_name2> <...>] " << endl << endl
            << "Calculate magnetic field along a magnetic field line, with given start position (x,y,z) and distance between steps (dist), up to max. number of steps (nsteps). " << endl
            << "Results are printed to terminal and saved to output file. Multiple magnetic field names can be specified, as defined in config file. " << endl;
        // output_file can be "-" (-> write to terminal)
        exit(-1);
    }

    mainmsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv, true);

    deque<string> tParameters = tXML.GetArguments().ParameterList();
    tParameters.pop_front();  // strip off config file name

    double tX0 = stod(tParameters[0]);
    double tY0 = stod(tParameters[1]);
    double tZ0 = stod(tParameters[2]);
    double tDistance = stod(tParameters[3]);
    uint32_t tNumSteps = stoi(tParameters[4]);

    string tOutFileName(tParameters[5]);
    ofstream tOutFileStream;
    streambuf* tOutFileBuf;
    if (tOutFileName == "-") {
        tOutFileBuf = cout.rdbuf();
    }
    else {
        tOutFileStream.open(tOutFileName);
        tOutFileBuf = tOutFileStream.rdbuf();
    }
    ostream tOutFile(tOutFileBuf);

    mainmsg(eNormal) << "...initialization finished" << eom;

    // initialize magnetic field
    KSRootMagneticField tRootMagneticField;

    for (size_t tIndex = 6; tIndex < tParameters.size(); tIndex++) {
        KSMagneticField* tMagneticFieldObject = getMagneticField(tParameters[tIndex]);
        tMagneticFieldObject->Initialize();
        tRootMagneticField.AddMagneticField(tMagneticFieldObject);
    }

    // intialize magnetic trajectory
    auto tMagneticTrajectory = new KSTrajTrajectoryMagnetic();
    auto tMagneticIntegrator = new KSTrajIntegratorRK8();
    auto tMagneticInterpolator = new KSTrajInterpolatorFast();
    auto tMagneticPropagation = new KSTrajTermPropagation();
    tMagneticPropagation->SetDirection(tDistance < 0 ? KSTrajTermPropagation::eBackward : KSTrajTermPropagation::eForward);
    auto tMagneticTimeStep = new KSTrajControlTime();
    tMagneticTimeStep->SetTime(tDistance);
    tMagneticTrajectory->SetIntegrator(tMagneticIntegrator);
    tMagneticTrajectory->SetInterpolator(tMagneticInterpolator);
    tMagneticTrajectory->AddTerm(tMagneticPropagation);
    tMagneticTrajectory->AddControl(tMagneticTimeStep);

    KThreeVector tMagneticField;
    KThreeVector tPosition(tX0, tY0, tZ0);

    uint32_t tStep = 0;
    while (tStep < tNumSteps) {
        try {
            tRootMagneticField.CalculateField(tPosition, 0.0, tMagneticField);
        }
        catch (...) {
            mainmsg(eWarning) << "error - cannot calculate field at position <" << tPosition << ">" << eom;
            continue;
        }

        mainmsg(eNormal) << "Magnetic Field at position " << tPosition << " is " << tMagneticField << eom;

        tOutFile << std::fixed << std::setprecision(16)
                 << tPosition.X() << "\t" << tPosition.Y() << "\t" << tPosition.Z() << "\t"
                 << tMagneticField.X() << "\t" << tMagneticField.Y() << "\t" << tMagneticField.Z() << endl;

        KSParticle tInitialParticle, tFinalParticle;
        KThreeVector tCenter;
        double tRadius, tTimeStep;

        tInitialParticle.SetMagneticFieldCalculator(&tRootMagneticField);
        tFinalParticle.SetMagneticFieldCalculator(&tRootMagneticField);

        tInitialParticle.SetPosition(tPosition);
        tInitialParticle.SetMagneticField(tMagneticField);

        if (tStep == 0 && tMagneticField.Z() < 0 && fabs(tMagneticField.Z()) > fabs(tMagneticField.X())+fabs(tMagneticField.Y())) {
            tMagneticPropagation->ReverseDirection();  // make sure distance > 0 tracks in positive z-direction
        }

        tMagneticTrajectory->CalculateTrajectory(tInitialParticle, tFinalParticle, tCenter, tRadius, tTimeStep);
        //tMagneticTrajectory->ExecuteTrajectory(tTimeStep, tFinalParticle);

        tPosition = tFinalParticle.GetPosition();
        tStep += 1;

        mainmsg(eDebug) << "Trajectory calculated next step: center=" << tCenter << " radius=" << tRadius << " timestep=" << tTimeStep << eom;
    }

    for (size_t tIndex = 6; tIndex < tParameters.size(); tIndex++) {
        KSMagneticField* tMagneticFieldObject = getMagneticField(tParameters[tIndex]);
        tMagneticFieldObject->Deinitialize();
    }

    if (tOutFileStream.is_open())
        tOutFileStream.close();

    return 0;
}
