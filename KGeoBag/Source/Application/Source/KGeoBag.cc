#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

using namespace KGeoBag;
using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 3) {
        cout << "usage: ./KGeoBag <config_file_name.xml> -r <NAME=VALUE> <NAME=VALUE> ..." << endl;
        return -1;
    }

    auto& tXML = KXMLInitializer::GetInstance();
    auto* tTokenizer = tXML.Configure(argc, argv);  // process extra files below

    //tXML.DumpConfiguration();

    coremsg(eNormal) << "starting..." << eom;

    auto tFileNames = tXML.GetArguments().ParameterList();
    tFileNames.pop_front();
    for (const auto& tFilename : tFileNames) {
        coremsg(eNormal) << "processing file <" << tFilename << "> ..." << eom;
        auto* tFile = new KTextFile();
        tFile->AddToNames(tFilename);
        tTokenizer->ProcessFile(tFile);
        delete tFile;
    }

    coremsg(eNormal) << "...finished" << eom;

    return 0;
}
