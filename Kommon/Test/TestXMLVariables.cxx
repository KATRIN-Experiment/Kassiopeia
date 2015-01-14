#include "KMessage.h"
#include "KTextFile.h"
#include "KXMLTokenizer.hh"
#include "KCommandLineTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KChattyProcessor.hh"

#include <cstdlib>

using namespace katrin;
using namespace katrin;

int main( int argc, char** argv )
{
    KMessageTable::GetInstance()->SetTerminalVerbosity( eNormal );
    KMessageTable::GetInstance()->SetLogVerbosity( eNormal );
    KTextFile* tFile = CreateConfigTextFile( "TestXMLVariables.xml" );

    KCommandLineTokenizer tCommandLine;
    tCommandLine.ProcessCommandLine( argc, argv );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor( tCommandLine.GetVariables() );
    KChattyProcessor tChattyProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tChattyProcessor.InsertAfter( &tVariableProcessor );

    tTokenizer.ProcessFile( tFile );

    return 0;
}




