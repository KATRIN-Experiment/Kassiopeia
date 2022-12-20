#include "KMessage.h"
#include "KTextFile.h"
#include "KToolbox.h"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    if (argc == 1) {
        cout
            << "usage: ./Serializer [--yaml|--xml] <ConfigFileName.xml> [ -r variable1=value1 variable2=value ... ]"
            << "Process a Kasper XML file and dump the resulting configuration in XML or YAML format."
            << endl;
        exit(-1);
    }

    KSerializationProcessor::EConfigFormat format = KSerializationProcessor::EConfigFormat::XML;
    if (argc >= 2 && (string(argv[1]) == string("--yaml"))) {
        //argv[1][0] = '\0';  // remove argument
        format = KSerializationProcessor::EConfigFormat::YAML;
    }
    else if (argc >= 2 && (string(argv[1]) == string("--json"))) {
        //argv[1][0] = '\0';  // remove argument
        format = KSerializationProcessor::EConfigFormat::JSON;
    }

    KMessageTable::GetInstance().SetTerminalVerbosity(eDebug);
    KMessageTable::GetInstance().SetLogVerbosity(eDebug);

    auto& tXML = KXMLInitializer::GetInstance();
    tXML.Configure(argc, argv);

    tXML.DumpConfiguration(cout, false, format);

    return 0;
}
