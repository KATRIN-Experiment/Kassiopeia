#include "KMessage.h"
#include "KTextFile.h"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KChattyProcessor.hh"

#include <cstdlib>

using namespace katrin;
using namespace katrin;

int main()
{
    KMessageTable::GetInstance()->SetTerminalVerbosity( eDebug );
    KMessageTable::GetInstance()->SetLogVerbosity( eDebug );
    KTextFile* tFile = CreateConfigTextFile( "TestXMLLoops.xml" );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KChattyProcessor tChattyProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tIncludeProcessor.InsertAfter( &tVariableProcessor );
    tLoopProcessor.InsertAfter( &tIncludeProcessor );
    tChattyProcessor.InsertAfter( &tLoopProcessor );

    tTokenizer.ProcessFile( tFile );

    return 0;
}




