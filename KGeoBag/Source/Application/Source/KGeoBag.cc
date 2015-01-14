#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"

#include "KMessage.h"
#include "KTextFile.h"
#include "KCommandLineTokenizer.hh"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KConditionProcessor.hh"
#include "KElementProcessor.hh"
#include "KTagProcessor.hh"

#ifdef KGEOBAG_USE_ROOT
#include "KFormulaProcessor.hh"
#endif

using namespace KGeoBag;
using namespace katrin;

int main( int argc, char** argv )
{
    if( argc < 3 )
    {
        cout << "usage: ./KGeoBag <config_file_name.xml> -r <NAME=VALUE> <NAME=VALUE> ..." << endl;
        return -1;
    }

    KCommandLineTokenizer tCommandLine;
    tCommandLine.ProcessCommandLine( argc, argv );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor( tCommandLine.GetVariables() );
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KConditionProcessor tConditionProcessor;
    KTagProcessor tTagProcessor;
    KElementProcessor tElementProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tIncludeProcessor.InsertAfter( &tVariableProcessor );

#ifdef KGEOBAG_USE_ROOT
	KFormulaProcessor tFormulaProcessor;
	tFormulaProcessor.InsertAfter( &tVariableProcessor );
	tIncludeProcessor.InsertAfter( &tFormulaProcessor );
#endif

    tLoopProcessor.InsertAfter( &tIncludeProcessor );
    tConditionProcessor.InsertAfter( &tLoopProcessor );
    tTagProcessor.InsertAfter( &tConditionProcessor );
    tElementProcessor.InsertAfter( &tTagProcessor );

    coremsg( eNormal ) << "starting..." << eom;

    KTextFile* tFile;
    for( vector< string >::const_iterator tIter = tCommandLine.GetFiles().begin(); tIter != tCommandLine.GetFiles().end(); tIter++ )
    {
        tFile = new KTextFile();
        tFile->AddToNames( *tIter );
        tTokenizer.ProcessFile( tFile );
        delete tFile;
    }

    coremsg( eNormal ) << "...finished" << eom;

    return 0;
}
