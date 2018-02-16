// Author: Daniel Hilk
// Date: 15.09.2017

#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <sys/stat.h>

#include "KTypelist.hh"
#include "KSurfaceContainer.hh"
#include "KEMFileInterface.hh"
#include "KBinaryDataStreamer.hh"

#include "KSADataStreamer.hh"
#include "KSerializer.hh"
#include "KSADataStreamer.hh"

using namespace KEMField;

int main(int argc, char* argv[])
{

    std::string usage =
    "\n"
    "Usage: KbdConverter <sbd input file without ending>\n"
    "\n"
    "This program takes a KEMField sbd file as input and writes a kbd file.\n"
    "\n";

    if( argc < 2 ) {
        std::cout << "Missing parameters. Aborting." << std::endl;
        exit(0);
    }

    std::cout << usage << std::endl;

    std::string tInFileName( argv[ 1 ] );  /* kbd output file */

    KSurfaceContainer container;

//    KMetadataStreamer mDS;
//    mDS.open(tInFileName + std::string(".smd"),"READ");
//    mDS >> container;
//    mDS.close();

    KBinaryDataStreamer bDS;
    bDS.open(tInFileName + std::string(".sbd"),"READ");
    bDS >> container;
    bDS.close();

//    KSADataStreamer saS;
//    saS.open(tInFileName + std::string(".ksa"),"READ");
//    saS >> container;
//    saS.close();

    std::cout << "Read in " << container.size() << " elements successfully." << std::endl;

    KEMFileInterface::GetInstance()->Write(container,"surfaceContainer");
    std::cout << "Kbd file " << KEMFileInterface::GetInstance()->GetActiveFileName() << " written successfully." << std::endl;

    return 0;
}
