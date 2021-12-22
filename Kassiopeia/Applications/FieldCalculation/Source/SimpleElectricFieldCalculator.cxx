#include "KMessage.h"
#include "KSFieldFinder.h"
#include "KSMainMessage.h"
#include "KSRootElectricField.h"
#include "KTextFile.h"
#include "KThreeVector.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#endif


using namespace Kassiopeia;
using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 6) {
        cout
            << "usage: ./SimpleElectricFieldCalculator <config_file.xml> <x> <y> <z> <electric_field_name1> [<electric_field_name2> <...>] "
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
    KSRootElectricField tRootElectricField;

    for (size_t tIndex = 3; tIndex < tParameters.size(); tIndex++) {
        KSElectricField* tElectricFieldObject = getElectricField(tParameters[tIndex]);
        tElectricFieldObject->Initialize();
        tRootElectricField.AddElectricField(tElectricFieldObject);
    }

    KThreeVector tElectricField;
    double tPotential;

    try {
        //tRootElectricField.CalculateFieldAndPotential( tPosition, 0., tElectricField, tPotential );  // FIXME: do not use this (issue #144)
        tRootElectricField.CalculateField(tPosition, 0.0, tElectricField);
        tRootElectricField.CalculatePotential(tPosition, 0.0, tPotential);
    }
    catch (...) {
        //cout << endl;
        mainmsg(eWarning) << "error - cannot calculate field at position <" << tPosition << ">" << eom;
        return 1;
    }

    mainmsg(eNormal) << "Electric Field at position " << tPosition << " is " << tElectricField << " and potential is "
                     << tPotential << eom;

    for (size_t tIndex = 3; tIndex < tParameters.size(); tIndex++) {
        KSElectricField* tElectricFieldObject = getElectricField(tParameters[tIndex]);
        tElectricFieldObject->Deinitialize();
    }

    return 0;
}
