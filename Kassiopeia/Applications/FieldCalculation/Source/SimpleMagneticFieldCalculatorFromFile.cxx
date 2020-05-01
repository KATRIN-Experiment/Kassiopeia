#include "KMessage.h"
#include "KSFieldFinder.h"
#include "KSMainMessage.h"
#include "KSRootMagneticField.h"
#include "KTextFile.h"
#include "KThreeVector.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>


using namespace Kassiopeia;
using namespace katrin;
using namespace KGeoBag;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 6) {
        cout
            << "usage: ./SimpleMagneticFieldCalculatorFromFile <config_file.xml> <input_file.txt> <output_file.txt> <number_of_lines> <magnetic_field_name1> [<magnetic_field_name2> <...>] "
            << endl;
        // output_file can be "-" (-> write to terminal)
        // number_of_lines can be negative (-> process all lines)
        exit(-1);
    }

    KMessageTable::GetInstance().SetPrecision(16);

    mainmsg(eNormal) << "starting initialization..." << eom;

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.AddDefaultIncludePath(CONFIG_DEFAULT_DIR);
    tXML.Configure(argc, argv, true);

    deque<string> tParameters = tXML.GetArguments().ParameterList();
    tParameters.pop_front();  // strip off config file name

    string inFileName(tParameters[0]);
    ifstream inFile;
    inFile.open(inFileName);

    string outFileName(tParameters[1]);
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

    istringstream tConverter(tParameters[2]);
    unsigned int noOfLines;
    tConverter >> noOfLines;

    mainmsg(eNormal) << "...initialization finished" << eom;

    // initialize magnetic field
    KSRootMagneticField tRootMagneticField;

    for (size_t tIndex = 3; tIndex < tParameters.size(); tIndex++) {
        KSMagneticField* tMagneticFieldObject = getMagneticField(tParameters[tIndex]);
        tMagneticFieldObject->Initialize();
        tRootMagneticField.AddMagneticField(tMagneticFieldObject);
    }

    KThreeVector tField(0., 0., 0.);

    for (unsigned int tLine = 0; tLine < noOfLines && inFile.good(); tLine++) {
        double x;
        double y;
        double z;

        inFile >> x;
        inFile >> y;
        inFile >> z;

        KThreeVector tPosition(x, y, z);
        cout << "calculating " << tPosition << " at line <" << tLine
             << "> ...                    \r";  // extra spaces needed!

        try {
            tRootMagneticField.CalculateField(tPosition, 0., tField);
        }
        catch (...) {
            //cout << endl;
            mainmsg(eWarning) << "error processing line <" << tLine << "> - cannot calculate field at position <"
                              << tPosition << ">" << eom;
            continue;
        }

        outFile << std::fixed << std::setprecision(16) << tPosition.X() << "\t" << tPosition.Y() << "\t"
                << tPosition.Z() << "\t" << tField.X() << "\t" << tField.Y() << "\t" << tField.Z() << endl;
    }

    for (size_t tIndex = 3; tIndex < tParameters.size(); tIndex++) {
        KSMagneticField* tMagneticFieldObject = getMagneticField(tParameters[tIndex]);
        tMagneticFieldObject->Deinitialize();
    }

    if (outFileStream.is_open())
        outFileStream.close();

    return 0;
}
