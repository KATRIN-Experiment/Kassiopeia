#include "KMessage.h"
#include "KTextFile.h"
#include "KXMLTokenizer.hh"
#include "KChattyProcessor.hh"

using namespace katrin;

int main()
{
    KMessageTable::GetInstance()->SetTerminalVerbosity( eDebug );
    KMessageTable::GetInstance()->SetLogVerbosity( eDebug );
    KTextFile* tFile = CreateConfigTextFile( "TestXMLTokenizer.xml" );

    KXMLTokenizer tTokenizer;
    KChattyProcessor tChattyProcessor;

    tChattyProcessor.InsertAfter( &tTokenizer );
    tTokenizer.ProcessFile( tFile );

    return 0;
}
