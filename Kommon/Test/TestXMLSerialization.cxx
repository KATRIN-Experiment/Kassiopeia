#include "KMessage.h"
#include "KTextFile.h"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KSerializationProcessor.hh"

#include <cstdlib>

using namespace katrin;
using namespace katrin;

int main()
{
    KMessageTable::GetInstance()->SetTerminalVerbosity( eDebug );
    KMessageTable::GetInstance()->SetLogVerbosity( eDebug );
    KTextFile* tFile = CreateConfigTextFile( "TestXMLSerialization.xml" );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KSerializationProcessor tKSerializationProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tKSerializationProcessor.InsertAfter( &tVariableProcessor );

    tTokenizer.ProcessFile( tFile );

    return 0;
}




