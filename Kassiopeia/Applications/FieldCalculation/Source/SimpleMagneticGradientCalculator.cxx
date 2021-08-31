#include "KMessage.h"
#include "KSFieldFinder.h"
#include "KSMainMessage.h"
#include "KSRootMagneticField.h"
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
    if (argc < 6) {
        cout
            << "usage: ./SimpleMagneticGradientCalculator <config_file.xml> <x> <y> <z> <magnetic_field_name1> [<magnetic_field_name2> <...>] "
            << endl;
        exit(-1);
    }

    KMessageTable::GetInstance().SetPrecision(16);

    mainmsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv, true);

    deque<string> tParameters = tXML.GetArguments().ParameterList();
    tParameters.pop_front();  // strip off config file name

    string tX(tParameters[0]);
    string tY(tParameters[1]);
    string tZ(tParameters[2]);
    string tSpaceString(" ");
    string tCombine = tX + tSpaceString + tY + tSpaceString + tZ;
    istringstream Converter(tCombine);
    KThreeVector tPosition;
    Converter >> tPosition;

    mainmsg(eNormal) << "...initialization finished" << eom;

    // initialize magnetic field
    KSRootMagneticField tRootMagneticField;

    for (size_t tIndex = 3; tIndex < tParameters.size(); tIndex++) {
        KSMagneticField* tMagneticFieldObject = getMagneticField(tParameters[tIndex]);
        tMagneticFieldObject->Initialize();
        tRootMagneticField.AddMagneticField(tMagneticFieldObject);
    }

    KThreeMatrix tMagneticGradient;

    try {
        tRootMagneticField.CalculateGradient(tPosition, 0.0, tMagneticGradient);
    }
    catch (...) {
        //cout << endl;
        mainmsg(eWarning) << "error - cannot calculate gradient at position <" << tPosition << ">" << eom;
        return 1;
    }

    mainmsg(eNormal) << "Magnetic Gradient at position " << tPosition << " is " << tMagneticGradient << eom;

    for (size_t tIndex = 3; tIndex < tParameters.size(); tIndex++) {
        KSMagneticField* tMagneticFieldObject = getMagneticField(tParameters[tIndex]);
        tMagneticFieldObject->Deinitialize();
    }

    return 0;
}
