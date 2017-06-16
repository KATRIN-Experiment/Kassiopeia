#include <KSSimulation.h>
#include <KSRoot.h>
#include "KMessage.h"
#include "KTextFile.h"

#include "KCommandLineTokenizer.hh"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KConditionProcessor.hh"
#include "KPrintProcessor.hh"
#include "KElementProcessor.hh"
#include "KTagProcessor.hh"

#ifdef Kassiopeia_USE_ROOT
#include "KFormulaProcessor.hh"
#endif

#include "KSMainMessage.h"
#include "KToolbox.h"

#include <vector>
#include <string>

using namespace Kassiopeia;
using namespace katrin;

int main( int argc, char** argv )
{
    if( argc == 1 )
    {
        cout << "usage: ./Kassiopeia <config_file_one.xml> [<config_file_one.xml> <...>] [ -r variable1=value1 variable2=value ... ]" << endl;
        exit( -1 );
    }

    KCommandLineTokenizer tCommandLine;
    tCommandLine.ProcessCommandLine( argc, argv );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor( tCommandLine.GetVariables() );
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KConditionProcessor tConditionProcessor;
    KPrintProcessor tPrintProcessor;
    KTagProcessor tTagProcessor;
    KElementProcessor tElementProcessor;

	tVariableProcessor.InsertAfter( &tTokenizer );
	tIncludeProcessor.InsertAfter( &tVariableProcessor );

#ifdef Kassiopeia_USE_ROOT
	KFormulaProcessor tFormulaProcessor;
	tFormulaProcessor.InsertAfter( &tVariableProcessor );
	tIncludeProcessor.InsertAfter( &tFormulaProcessor );
#endif

    tLoopProcessor.InsertAfter( &tIncludeProcessor );
    tConditionProcessor.InsertAfter( &tLoopProcessor );
    tPrintProcessor.InsertAfter( &tConditionProcessor );
    tTagProcessor.InsertAfter( &tPrintProcessor );

    tElementProcessor.InsertAfter( &tTagProcessor );

    mainmsg( eNormal ) << "starting..." << eom;

    KToolbox::GetInstance();

    KTextFile* tFile;
    for( auto tFilename : tCommandLine.GetFiles() )
    {
        tFile = new KTextFile();
        tFile->AddToNames( tFilename );
        tTokenizer.ProcessFile( tFile );
        delete tFile;
    }
/*
    KSRoot* tRoot = KToolbox::GetInstance().GetAll<KSRoot>()[0];
    for( auto sim : KToolbox::GetInstance().GetAll<KSSimulation>())
    {
        tRoot->Execute(sim);
    }
*/

    mainmsg( eNormal ) << "...finished" << eom;

    return 0;
}
