#include "KMessage.h"
#include "KTextFile.h"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KPrintProcessor.hh"

#include <cstdlib>

using namespace katrin;
using namespace katrin;

int main()
{
    KMessageTable::GetInstance()->SetTerminalVerbosity( eDebug );
    KMessageTable::GetInstance()->SetLogVerbosity( eDebug );
    KTextFile* tFile = CreateConfigTextFile( "TestXMLPrint.xml" );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KFormulaProcessor tFormulaProcessor;
    KPrintProcessor tPrintProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tFormulaProcessor.InsertAfter( &tVariableProcessor );
    tPrintProcessor.InsertAfter( &tFormulaProcessor );

    tTokenizer.ProcessFile( tFile );

    return 0;
}




