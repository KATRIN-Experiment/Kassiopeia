#include "KMessage.h"
#include "KTextFile.h"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KConditionProcessor.hh"
#include "KChattyProcessor.hh"

using namespace katrin;

int main()
{
    KMessageTable::GetInstance()->SetTerminalVerbosity( eDebug );
    KMessageTable::GetInstance()->SetLogVerbosity( eDebug );
    KTextFile* tFile = CreateConfigTextFile( "TestXMLConditions.xml" );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KFormulaProcessor tFormulaProcessor;
    KConditionProcessor tConditionProcessor;
    KIncludeProcessor tIncludeProcessor;
    KChattyProcessor tChattyProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tFormulaProcessor.InsertAfter( &tVariableProcessor );
    tConditionProcessor.InsertAfter( &tFormulaProcessor );
    tIncludeProcessor.InsertAfter( &tConditionProcessor );
    tChattyProcessor.InsertAfter( &tIncludeProcessor );

    tTokenizer.ProcessFile( tFile );

    return 0;
}




