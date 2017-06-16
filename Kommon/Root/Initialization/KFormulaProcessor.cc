#include "KFormulaProcessor.hh"
#include "KInitializationMessage.hh"

#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <memory>

#include <TFormula.h>

using namespace std;

namespace katrin
{

    const string KFormulaProcessor::fStartBracket = string( "{" );
    const string KFormulaProcessor::fEndBracket = string( "}" );
    const string KFormulaProcessor::fEqual = string( "eq" );
    const string KFormulaProcessor::fNonEqual = string( "ne" );
    const string KFormulaProcessor::fGreater = string( "gt" );
    const string KFormulaProcessor::fLess = string( "lt" );
    const string KFormulaProcessor::fGreaterEqual = string( "ge" );
    const string KFormulaProcessor::fLessEqual = string( "le" );
    const string KFormulaProcessor::fModulo = string( "mod" );


    KFormulaProcessor::KFormulaProcessor()
    {
    }

    KFormulaProcessor::~KFormulaProcessor()
    {
    }

    void KFormulaProcessor::ProcessToken( KAttributeDataToken* aToken )
    {
        Evaluate( aToken );
        KProcessor::ProcessToken( aToken );
        return;
    }

    void KFormulaProcessor::ProcessToken( KElementDataToken* aToken )
    {
        Evaluate( aToken );
        KProcessor::ProcessToken( aToken );
        return;
    }

    void KFormulaProcessor::Evaluate( KToken* aToken )
    {
        string tValue;
        string tBuffer;
        stack< string > tBufferStack;

        stringstream tResultConverter;

        tValue = aToken->GetValue();

        tBufferStack.push( "" );
        for( size_t Index = 0; Index < tValue.size(); Index++ )
        {
            if( tValue[Index] == fStartBracket[0] )
            {
                tBufferStack.top() += tBuffer;
                tBufferStack.push( "" );
                tBuffer.clear();
                continue;
            }

            if( tValue[Index] == fEndBracket[0] )
            {
                tBufferStack.top() += tBuffer;
                tBuffer = tBufferStack.top();
                tBufferStack.pop();
                if( tBufferStack.size() == 0 )
                {
                    initmsg( eError ) << "bracket matching problem at position <" << Index << "> in string <" << tValue << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    return;
                }

                //conversions for logical operations
                while ( tBuffer.find( fGreaterEqual ) != string::npos )
                {
                	tBuffer.replace( tBuffer.find( fGreaterEqual ), fGreaterEqual.length(), string(">=") );
                }
                while ( tBuffer.find( fLessEqual ) != string::npos )
                {
                	tBuffer.replace( tBuffer.find( fLessEqual ), fLessEqual.length(), string("<=") );
                }
                while ( tBuffer.find( fNonEqual ) != string::npos )
                {
                	tBuffer.replace( tBuffer.find( fNonEqual ), fNonEqual.length(), string("!=") );
                }
                while ( tBuffer.find( fEqual ) != string::npos )
                {
                	tBuffer.replace( tBuffer.find( fEqual ), fEqual.length(), string("==") );
                }
                while ( tBuffer.find( fGreater ) != string::npos )
                {
                	tBuffer.replace( tBuffer.find( fGreater ), fGreater.length(), string(">") );
                }
                while ( tBuffer.find( fLess ) != string::npos )
                {
                	tBuffer.replace( tBuffer.find( fLess ), fLess.length(), string("<") );
                }

                while ( tBuffer.find( fModulo ) != string::npos )
                {
                	tBuffer.replace( tBuffer.find( fModulo ), fModulo.length(), string("%") );
                }

                TFormula formulaParser("(anonymous)", tBuffer.c_str());
                tResultConverter.str("");
                tResultConverter << std::setprecision( 15 ) << formulaParser.Eval( 0.0 );
                tBuffer = tResultConverter.str();

                tBufferStack.top() += tBuffer;
                tBuffer.clear();
                continue;
            }

            tBuffer.append( 1, tValue[Index] );
        }
        tBufferStack.top() += tBuffer;
        tValue = tBufferStack.top();
        tBufferStack.pop();

        if( tBufferStack.size() != 0 )
        {
            initmsg( eError ) << "bracket matching problem at end of string <" << tValue << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            return;
        }

        aToken->SetValue( tValue );

        return;
    }

} /* namespace Kassiopeia */
