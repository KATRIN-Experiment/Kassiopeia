#include "KMessage.h"
#include "KTextFile.h"
#include "KToolbox.h"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"

using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    if (argc == 1) {
        cout
            << "usage: ./VariableProcessor <ConfigFileName.xml> [ -r variable1=value1 variable2=value ... ]"
            << "Process a Kasper XML file and dump the defined external/global variables in JSON format."
            << endl;
        exit(-1);
    }

    KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    KMessageTable::GetInstance().SetLogVerbosity(eDebug);

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.Configure(argc, argv);

    std::cout << "{" << std::endl;
    std::cout << "  \"external\": {" << std::endl;

    const auto& externalVars = tXML.GetVariableProcessor()->GetExternalVariables();
    for (auto & it : externalVars) {
        std::cout << "    \"" << it.first << "\": \"" << it.second << "\"";
        if (it != *(externalVars.rbegin()))
            std::cout << ",";
        std::cout << std::endl;
    }

    std::cout << "  }," << std::endl;
    std::cout << "  \"global\": {" << std::endl;

    const auto& globalVars = tXML.GetVariableProcessor()->GetGlobalVariables();
    for (auto & it : globalVars) {
        std::cout << "    \"" << it.first << "\": \"" << it.second << "\"";
        if (it != *(globalVars.rbegin()))
            std::cout << ",";
        std::cout << std::endl;
    }

    std::cout << "  }" << std::endl;
    std::cout << "}" << std::endl;

    return 0;
}
