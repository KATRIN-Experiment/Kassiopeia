#include "KMessage.h"
#include "KRandom.h"
#include "KSFieldFinder.h"
#include "KSMainMessage.h"
#include "KThreeVector.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

#ifdef KEMFIELD_USE_PETSC
#include "KPETScInterface.hh"
#elif KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#endif

// timing function
#include <ctime>
#include <sys/time.h>

/* Remove if already defined */
using int64 = long long;
using uint64 = unsigned long long;

/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
 * windows and linux. */

uint64 GetTimeMs64()
{
    /* Linux */
    struct timeval tv;

    gettimeofday(&tv, nullptr);

    uint64 ret = tv.tv_usec;
    /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
    ret /= 1000;

    /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
    ret += (tv.tv_sec * 1000);

    return ret;
}


using namespace Kassiopeia;
using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 4) {
        cout
            << "usage: ./SimpleElectricFieldCalculatorSpeedTest <config_file.xml> <N> <electric_field_name1> [<electric_field_name2> <...>] "
            << endl;
        exit(-1);
    }

#ifdef KEMFIELD_USE_PETSC
    KEMField::KPETScInterface::GetInstance()->Initialize(&argc, &argv);
#elif KEMFIELD_USE_MPI
    KEMField::KMPIInterface::GetInstance()->Initialize(&argc, &argv);
#endif

    mainmsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv, true);

    deque<string> tParameters = tXML.GetArguments().ParameterList();
    tParameters.pop_front();  // strip off config file name


    istringstream Converter(tParameters[0]);
    long tNumber;
    Converter >> tNumber;

    mainmsg(eNormal) << "...initialization finished" << eom;

    // initialize electric field
    vector<KSElectricField*> tElectricFields;

    for (size_t tIndex = 1; tIndex < tParameters.size(); tIndex++) {
        KSElectricField* tElectricFieldObject = getElectricField(tParameters[tIndex]);
        tElectricFieldObject->Initialize();
        tElectricFields.push_back(tElectricFieldObject);
    }

    KThreeVector tElectricField;
    double tPotential;

    double tMaxDistance = 5.0;
    double tInnerRadius = 3.0;
    vector<KThreeVector> tPositions;
    tPositions.resize(tNumber);
    for (long tIndex = 0; tIndex < tNumber; tIndex++) {
        double r = katrin::KRandom::GetInstance().Uniform(0.0, tInnerRadius);
        double phi = katrin::KRandom::GetInstance().Uniform(0, 360);
        double z = katrin::KRandom::GetInstance().Uniform(-tMaxDistance, tMaxDistance);

        double x = r * cos(phi);
        double y = r * sin(phi);
        tPositions.emplace_back(x, y, z);
    }

    for (auto& tFieldObject : tElectricFields) {
        uint64 tStartTime = GetTimeMs64();

        for (size_t tIndex = 0; tIndex < tPositions.size(); tIndex++) {
            try {
                //tRootElectricField.CalculateFieldAndPotential( tPosition, 0., tField, tPotential );  // FIXME: do not use this (issue #144)
                tFieldObject->CalculateField(tPositions.at(tIndex), 0.0, tElectricField);
                tFieldObject->CalculatePotential(tPositions.at(tIndex), 0.0, tPotential);
            }
            catch (...) {
                mainmsg(eWarning) << "error processing index <" << tIndex << "> - cannot calculate field at position <"
                                  << tPositions.at(tIndex) << ">" << eom;
                continue;
            }
        }

        uint64 tEndTime = GetTimeMs64();
        mainmsg << "Elapsed time for field <" << tFieldObject->GetName() << "> is <" << tEndTime - tStartTime << "> ms "
                << eom;
        mainmsg << "Position of last field point is " << tPositions.at(tPositions.size() - 1) << eom;
        mainmsg << "Value of last field point is " << tElectricField << eom;
        mainmsg << "Value of last potential is " << tPotential << eom;
    }


    for (size_t tIndex = 1; tIndex < tParameters.size(); tIndex++) {
        KSElectricField* tElectricFieldObject = getElectricField(tParameters[tIndex]);
        tElectricFieldObject->Deinitialize();
    }

    return 0;
}
