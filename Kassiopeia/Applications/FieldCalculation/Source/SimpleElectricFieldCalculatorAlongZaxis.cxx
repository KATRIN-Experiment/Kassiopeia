#include "KMessage.h"
#include "KSFieldFinder.h"
#include "KSMainMessage.h"
#include "KSRootElectricField.h"
#include "KTextFile.h"
#include "KThreeVector.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"


using namespace Kassiopeia;
using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 6) {
        cout
            << "usage: ./SimpleElectricFieldCalculatorAlongZaxis <config_file.xml> <z1> <z2> <dz> <output_file.txt> <electric_field_name1> [<electric_field_name2> <...>] "
            << endl;
        // output_file can be "-" (-> write to terminal)
        exit(-1);
    }

    mainmsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv, true);

    deque<string> tParameters = tXML.GetArguments().ParameterList();
    tParameters.pop_front();  // strip off config file name

    double Z1 = stod(tParameters[0]);
    double Z2 = stod(tParameters[1]);
    double dZ = stod(tParameters[2]);

    string outFileName(tParameters[3]);
    ofstream outFileStream;
    streambuf* outFileBuf;
    if (outFileName == "-") {
        outFileBuf = cout.rdbuf();
    }
    else {
        outFileStream.open(outFileName);
        outFileBuf = outFileStream.rdbuf();
    }
    ostream outFile(outFileBuf);

    mainmsg(eNormal) << "...initialization finished" << eom;

    // initialize Electric field
    KSRootElectricField tRootElectricField;

    for (size_t tIndex = 4; tIndex < tParameters.size(); tIndex++) {
        KSElectricField* tElectricFieldObject = getElectricField(tParameters[tIndex]);
        tElectricFieldObject->Initialize();
        tRootElectricField.AddElectricField(tElectricFieldObject);
    }

    KThreeVector tElectricField;
    double tPotential;

    for (double z = Z1; z <= Z2; z += dZ) {
        KThreeVector tPosition(0, 0, z);

        try {
            //tRootElectricField.CalculateFieldAndPotential( tPosition, 0., tField, tPotential );  // FIXME: do not use this (issue #144)
            tRootElectricField.CalculateField(tPosition, 0.0, tElectricField);
            tRootElectricField.CalculatePotential(tPosition, 0.0, tPotential);
        }
        catch (...) {
            //cout << endl;
            mainmsg(eWarning) << "error - cannot calculate field at position <" << tPosition << ">" << eom;
            continue;
        }

        mainmsg(eNormal) << "Electric Field at position " << tPosition << " is " << tElectricField
                         << " and potential is " << tPotential << eom;

        outFile << std::fixed << std::setprecision(16) << tPosition.X() << "\t" << tPosition.Y() << "\t"
                << tPosition.Z() << "\t" << tPotential << "\t" << tElectricField.X() << "\t" << tElectricField.Y()
                << "\t" << tElectricField.Z() << endl;
    }

    for (size_t tIndex = 4; tIndex < tParameters.size(); tIndex++) {
        KSElectricField* tElectricFieldObject = getElectricField(tParameters[tIndex]);
        tElectricFieldObject->Deinitialize();
    }

    if (outFileStream.is_open())
        outFileStream.close();

    return 0;
}
