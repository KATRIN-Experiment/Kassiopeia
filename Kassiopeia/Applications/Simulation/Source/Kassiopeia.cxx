#include "KSMainMessage.h"
#include "KSSimulation.h"
//#include "KSRoot.h"

#include "KMessage.h"
#include "KTextFile.h"
#include "KToolbox.h"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

#include <string>
#include <vector>

using namespace Kassiopeia;
using namespace katrin;

int main(int argc, char** argv)
{
    if (argc == 1) {
        cout
            << "usage: ./Kassiopeia <config_file_one.xml> [<config_file_one.xml> <...>] [ -r variable1=value1 variable2=value ... ]"
            << endl;
        exit(-1);
    }

    auto& tXML = KXMLInitializer::GetInstance();
    auto* tTokenizer = tXML.Configure(argc, argv, false);  // process files below

    //tXML.DumpConfiguration();

    mainmsg(eNormal) << "starting..." << eom;

    KToolbox::GetInstance();

    for (auto tFilename : tXML.GetArguments().ParameterList()) {
        mainmsg(eInfo) << "processing file <" << tFilename << "> ..." << eom;
        auto* tFile = new KTextFile();
        tFile->AddToNames(tFilename);
        tTokenizer->ProcessFile(tFile);
        delete tFile;
    }
    /*
        KSRoot* tRoot = KToolbox::GetInstance().GetAll<KSRoot>()[0];
        for( auto sim : KToolbox::GetInstance().GetAll<KSSimulation>())
        {
            tRoot->Execute(sim);
        }
    */

    mainmsg(eNormal) << "...finished" << eom;

    return 0;
}
