#include "KESSScatteringCalculator.h"

#include "KConst.h"
#include "KFile.h"
#include "KSInteractionsMessage.h"
#include "KTextFile.h"
#include "KThreeVector.hh"

#include <map>

using namespace std;
using namespace katrin;
using namespace KGeoBag;

namespace Kassiopeia
{

KESSScatteringCalculator::KESSScatteringCalculator() :
    fInteraction("none"),
    fIonisationCalculator(nullptr),
    fRelaxationCalculator(nullptr)
{}

KESSScatteringCalculator::~KESSScatteringCalculator() {}


void KESSScatteringCalculator::ReadMFP(string data_filename, map<double, double>& MapForTables)
{
    char line[196];
    double one = 0, two = 0, three = 0;

    string myPathToTable = DATA_DEFAULT_DIR;

    string UnusedReturnValue;
    FILE* MapFile = fopen((myPathToTable + "/" + data_filename).c_str(), "r");

    if (MapFile == nullptr) {
        intmsg(eError) << "FILE " << myPathToTable + "/" + data_filename << " NOT FOUND!\n"
                       << ret << "CHECK YOUR IO CONFIGFILE FOR DATA DIRECTORY" << eom;
    }

    for (unsigned int i = 0; i < 2; i++) {
        UnusedReturnValue = fgets(line, 195, MapFile);
    }

    while (fgets(line, 195, MapFile) != nullptr) {
        sscanf(line, "%lf %lf %lf", &one, &two, &three);
        if (feof(MapFile) == false) {
            MapForTables.insert(pair<double, double>(one, two));
        }
    }
}

void KESSScatteringCalculator::ReadPDF(string data_filename, map<double, vector<vector<double>>>& MapForTables)
{
    char line[196];
    double one = 0, two = 0, three = 0;

    string myPathToTable = DATA_DEFAULT_DIR;

    string UnusedReturnValue;

    double oldOne = 0.;

    vector<vector<double>> elScTable;
    vector<double> theta;          //
    vector<double> integralTheta;  //

    //change filename here
    FILE* elasticTable = fopen((myPathToTable + "/" + data_filename).c_str(), "r");

    if (elasticTable == nullptr) {
        intmsg(eError) << "FILE " << myPathToTable + "/" + data_filename << " NOT FOUND!\n"
                       << ret << "CHECK YOUR IO CONFIGFILE FOR DATA DIRECTORY" << eom;
    }
    for (unsigned int i = 0; i < 3; i++) {
        UnusedReturnValue = fgets(line, 195, elasticTable);
    }
    while (fgets(line, 195, elasticTable) != nullptr) {
        sscanf(line, "%lf %lf %lf", &one, &two, &three);
        if (feof(elasticTable) == false) {
            //if all values for one energy are read in
            //save the vectors to the big vector
            //make a entry in the map for faster searching
            //clear the value tables
            if (one != oldOne && oldOne > 0.) {
                elScTable.push_back(theta);
                elScTable.push_back(integralTheta);

                MapForTables.insert(pair<double, vector<vector<double>>>(oldOne, elScTable));

                theta.clear();
                integralTheta.clear();
                elScTable.clear();
            }
            theta.push_back(two);
            integralTheta.push_back(three);
        }
        oldOne = one;
    }
    fclose(elasticTable);
}
}  // namespace Kassiopeia

/* namespace Kassiopeia */
