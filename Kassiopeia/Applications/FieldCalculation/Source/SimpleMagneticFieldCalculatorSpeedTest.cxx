#include "KMessage.h"
#include "KRandom.h"
#include "KSFieldFinder.h"
#include "KSMainMessage.h"
#include "KSRootMagneticField.h"
#include "KTextFile.h"
#include "KThreeMatrix.hh"
#include "KThreeVector.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

// timing function
#include <ctime>
#include <sys/time.h>

/* Remove if already defined */
typedef long long int64;
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
using namespace KGeoBag;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 4) {
        cout
            << "usage: ./SimpleMagneticFieldCalculatorSpeedTest <config_file.xml> <N> <magnetic_field_name1> [<magnetic_field_name2> <...>] "
            << endl;
        exit(-1);
    }


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

    KMessageTable::GetInstance().SetPrecision(20);

    // initialize magnetic field
    vector<KSMagneticField*> tMagneticFields;

    for (size_t tIndex = 1; tIndex < tParameters.size(); tIndex++) {
        KSMagneticField* tMagneticFieldObject = getMagneticField(tParameters[tIndex]);
        tMagneticFieldObject->Initialize();
        tMagneticFields.push_back(tMagneticFieldObject);
    }

    KThreeVector tMagneticField;
    KThreeMatrix tMagneticFieldGradient;

    double tPinchCenter = 12.18375;
    double tMaxDistance = 5.0;
    double tInnerRadius = 0.227;
    vector<KThreeVector> tPositions;
    tPositions.resize(tNumber);
    for (long tIndex = 0; tIndex < tNumber; tIndex++) {
        double r = katrin::KRandom::GetInstance().Uniform(0.0, tInnerRadius);
        double phi = katrin::KRandom::GetInstance().Uniform(0, 360);
        double z = katrin::KRandom::GetInstance().Uniform(tPinchCenter - tMaxDistance, tPinchCenter + tMaxDistance);

        double x = r * cos(phi);
        double y = r * sin(phi);
        tPositions.emplace_back(x, y, z);
    }

    for (auto& tFieldObject : tMagneticFields) {
        uint64 tStartTime = GetTimeMs64();

        for (size_t tIndex = 0; tIndex < tPositions.size(); tIndex++) {
            try {
                tFieldObject->CalculateFieldAndGradient(tPositions.at(tIndex),
                                                        0.0,
                                                        tMagneticField,
                                                        tMagneticFieldGradient);
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
        mainmsg << "Value of last field point is " << tMagneticField << eom;
        mainmsg << "Value of last field gradient is " << tMagneticFieldGradient << eom;
    }


    for (size_t tIndex = 1; tIndex < tParameters.size(); tIndex++) {
        KSMagneticField* tMagneticFieldObject = getMagneticField(tParameters[tIndex]);
        tMagneticFieldObject->Deinitialize();
    }

    return 0;
}
