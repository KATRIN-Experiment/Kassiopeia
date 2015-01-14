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

#ifdef Kommon_USE_ROOT
#include "KFormulaProcessor.hh"
#endif

#include "KSMainMessage.h"
#include "KSToolbox.h"

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

#ifdef Kommon_USE_ROOT
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

    KSToolbox::GetInstance();

    KTextFile* tFile;
    for( vector< string >::const_iterator tIter = tCommandLine.GetFiles().begin(); tIter != tCommandLine.GetFiles().end(); tIter++ )
    {
        tFile = new KTextFile();
        tFile->AddToNames( *tIter );
        tTokenizer.ProcessFile( tFile );
        delete tFile;
    }

    KSToolbox::DeleteInstance();

    mainmsg( eNormal ) << "...finished" << eom;

    return 0;
}
