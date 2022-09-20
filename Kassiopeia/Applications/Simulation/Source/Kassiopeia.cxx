#include "KSMainMessage.h"

#include "KMessage.h"
#include "KTextFile.h"
#include "KToolbox.h"
#include "KXMLInitializer.hh"
#include "KXMLTokenizer.hh"

#ifdef KEMFIELD_USE_PETSC
#include "KPETScInterface.hh"
#elif KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#endif

#include <string>
#include <vector>

using namespace Kassiopeia;
using namespace katrin;

int main(int argc, char** argv)
{
    if (argc == 1) {
        std::cout
            << "usage: ./Kassiopeia <config_file_one.xml> [<config_file_two.xml> <...>]"
            << " [ -v | -q ] [ -b | -batch ]"
            << " [ -r variable1=value1 variable2=value ... ] [ --variable3=value3 ... ]"
            << std::endl;
        exit(-1);
    }

#ifdef KEMFIELD_USE_PETSC
    KEMField::KPETScInterface::GetInstance()->Initialize(&argc, &argv);
#elif KEMFIELD_USE_MPI
    KEMField::KMPIInterface::GetInstance()->Initialize(&argc, &argv);
#endif

    auto& tXML = KXMLInitializer::GetInstance();
    auto* tTokenizer = tXML.Configure(argc, argv);  // process extra files below

    //tXML.DumpConfiguration();

    mainmsg(eNormal) << "starting..." << eom;

    KToolbox::GetInstance();

    auto tFileNames = tXML.GetArguments().ParameterList();
    tFileNames.pop_front();
    for (const auto& tFilename : tFileNames) {
        mainmsg(eNormal) << "processing file <" << tFilename << "> ..." << eom;
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
