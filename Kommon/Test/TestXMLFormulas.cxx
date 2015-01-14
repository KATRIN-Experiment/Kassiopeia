#include "KMessage.h"
#include "KTextFile.h"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KChattyProcessor.hh"

#include <cstdlib>

using namespace katrin;
using namespace katrin;

int main()
{
    KMessageTable::GetInstance()->SetTerminalVerbosity( eDebug );
    KMessageTable::GetInstance()->SetLogVerbosity( eDebug );
    KTextFile* tFile = CreateConfigTextFile( "TestXMLFormulas.xml" );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KFormulaProcessor tFormulaProcessor;
    KChattyProcessor tChattyProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tFormulaProcessor.InsertAfter( &tVariableProcessor );
    tChattyProcessor.InsertAfter( &tFormulaProcessor );

    tTokenizer.ProcessFile( tFile );

    return 0;
}




