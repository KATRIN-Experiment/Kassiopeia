#include "KMessage.h"
#include "KTextFile.h"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KChattyProcessor.hh"

#include <cstdlib>

using namespace katrin;
using namespace katrin;

int main()
{
    KMessageTable::GetInstance()->SetTerminalVerbosity( eDebug );
    KMessageTable::GetInstance()->SetLogVerbosity( eDebug );
    KTextFile* tFile = CreateConfigTextFile( "TestXMLIncludes.xml" );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KIncludeProcessor tIncludeProcessor;
    KChattyProcessor tChattyProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tIncludeProcessor.InsertAfter( &tVariableProcessor );
    tChattyProcessor.InsertAfter( &tIncludeProcessor );

    tTokenizer.ProcessFile( tFile );

    return 0;
}
