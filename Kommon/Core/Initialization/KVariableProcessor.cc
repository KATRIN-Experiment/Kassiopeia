#include "KVariableProcessor.hh"

#include "KInitializationMessage.hh"

#include <cstdlib>

namespace katrin
{
    const string KVariableProcessor::fStartBracket = "[";
    const string KVariableProcessor::fEndBracket = "]";
    const string KVariableProcessor::fNameValueSeparator = ":";

    KVariableProcessor::KVariableProcessor() :
            KProcessor(),
            fElementState( eElementInactive ),
            fAttributeState( eAttributeInactive ),
            fName( "" ),
            fValue( "" ),
            fExternalMap( new VariableMap() ),
            fGlobalMap( new VariableMap() ),
            fLocalMap( new VariableMap() ),
            fLocalMapStack()
    {
    }
    KVariableProcessor::KVariableProcessor( const VariableMap& anExternalMap ) :
            KProcessor(),
            fElementState( eElementInactive ),
            fAttributeState( eAttributeInactive ),
            fName( "" ),
            fValue( "" ),
            fExternalMap( new VariableMap( anExternalMap ) ),
            fGlobalMap( new VariableMap() ),
            fLocalMap( new VariableMap() ),
            fLocalMapStack()
    {
    }

    KVariableProcessor::~KVariableProcessor()
    {
        delete fExternalMap;
        delete fGlobalMap;
        delete fLocalMap;
    }

    void KVariableProcessor::ProcessToken( KBeginFileToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            fLocalMapStack.push( fLocalMap );
            fLocalMap = new VariableMap();

            KProcessor::ProcessToken( aToken );
            return;
        }

        initmsg( eError ) << "got unknown start of file" << ret;
        initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
        return;
    }

    void KVariableProcessor::ProcessToken( KBeginElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            if( aToken->GetValue() == string( "define" ) )
            {
                fElementState = eActiveLocalDefine;
                return;
            }

            if( aToken->GetValue() == string( "global_define" ) )
            {
                fElementState = eActiveGlobalDefine;
                return;
            }

            if( aToken->GetValue() == string( "external_define" ) )
            {
                fElementState = eActiveExternalDefine;
                return;
            }

            if( aToken->GetValue() == string( "undefine" ) )
            {
                fElementState = eActiveLocalUndefine;
                return;
            }

            if( aToken->GetValue() == string( "global_undefine" ) )
            {
                fElementState = eActiveGlobalUndefine;
                return;
            }

            if( aToken->GetValue() == string( "external_undefine" ) )
            {
                fElementState = eActiveExternalUndefine;
                return;
            }

            KProcessor::ProcessToken( aToken );
            return;
        }

        initmsg( eError ) << "got unknown element <" << aToken->GetValue() << ">" << ret;
        initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
        return;
    }

    void KVariableProcessor::ProcessToken( KBeginAttributeToken* aToken )
    {
        if( (fElementState == eActiveLocalDefine) || (fElementState == eActiveGlobalDefine) || (fElementState == eActiveExternalDefine) )
        {
            if( aToken->GetValue() == "name" )
            {
                if( fName.size() == 0 )
                {
                    fAttributeState = eActiveName;
                    return;
                }
                else
                {
                    initmsg << "name attribute must appear only once in definition" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    return;
                }
            }

            if( aToken->GetValue() == "value" )
            {
                if( fValue.size() == 0 )
                {
                    fAttributeState = eActiveValue;
                    return;
                }
                else
                {
                    initmsg << "value attribute must appear only once in definition" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    return;
                }
            }

            initmsg( eError ) << "got unknown attribute <" << aToken->GetValue() << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;

            return;
        }
        if( (fElementState == eActiveLocalUndefine) || (fElementState == eActiveGlobalUndefine) || (fElementState == eActiveExternalUndefine) )
        {
            if( aToken->GetValue() == "name" )
            {
                if( fName.size() == 0 )
                {
                    fAttributeState = eActiveName;
                    return;
                }
                else
                {
                    initmsg( eError ) << "name attribute must appear only once in definition" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    return;
                }
            }

            initmsg( eError ) << "got unknown attribute <" << aToken->GetValue() << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;

            return;
        }

        KProcessor::ProcessToken( aToken );
        return;
    }

    void KVariableProcessor::ProcessToken( KAttributeDataToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            Evaluate( aToken );
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( (fElementState == eActiveLocalDefine) || (fElementState == eActiveGlobalDefine) || (fElementState == eActiveExternalDefine) || (fElementState == eActiveLocalUndefine) || (fElementState == eActiveGlobalUndefine) || (fElementState == eActiveExternalUndefine) )
        {
            if( fAttributeState == eActiveName )
            {
                Evaluate( aToken );
                fName = aToken->GetValue();
                fAttributeState = eAttributeComplete;
                return;
            }
            if( fAttributeState == eActiveValue )
            {
                Evaluate( aToken );
                fValue = aToken->GetValue();
                fAttributeState = eAttributeComplete;
                return;
            }
        }

        return;
    }

    void KVariableProcessor::ProcessToken( KEndAttributeToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( (fElementState == eActiveLocalDefine) || (fElementState == eActiveGlobalDefine) || (fElementState == eActiveExternalDefine) || (fElementState == eActiveLocalUndefine) || (fElementState == eActiveGlobalUndefine) || (fElementState == eActiveExternalUndefine) )
        {
            if( fAttributeState == eAttributeComplete )
            {
                fAttributeState = eAttributeInactive;
                return;
            }
        }

        return;
    }

    void KVariableProcessor::ProcessToken( KMidElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActiveLocalDefine )
        {
            VariableIt ExternalIt = fExternalMap->find( fName );
            if( ExternalIt == fExternalMap->end() )
            {
                VariableIt GlobalIt = fGlobalMap->find( fName );
                if( GlobalIt == fGlobalMap->end() )
                {
                    VariableIt FileIt = fLocalMap->find( fName );
                    if( FileIt == fLocalMap->end() )
                    {
                        fLocalMap->insert( VariableEntry( fName, fValue ) );
                    }
                    else
                    {
                        initmsg( eError ) << "redefinition of local variable  <" << fName << "> with new value <" << fValue << ">" << ret;
                        initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    }
                }
                else
                {
                    initmsg( eError ) << "redefinition of global variable  <" << fName << "> with new value <" << fValue << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }

            }
            else
            {
                initmsg( eError ) << "redefinition of external variable  <" << fName << "> with new value <" << fValue << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            }

            fElementState = eElementComplete;
            fName.clear();
            fValue.clear();
            return;
        }

        if( fElementState == eActiveGlobalDefine )
        {
            VariableIt ExternalIt = fExternalMap->find( fName );
            if( ExternalIt == fExternalMap->end() )
            {
                VariableIt LocalIt = fLocalMap->find( fName );
                if( LocalIt == fLocalMap->end() )
                {
                    VariableIt GlobalIt = fGlobalMap->find( fName );
                    if( GlobalIt == fGlobalMap->end() )
                    {

                        fGlobalMap->insert( VariableEntry( fName, fValue ) );
                    }
                    else
                    {
                        initmsg( eError ) << "redefinition of global variable  <" << fName << "> with new value <" << fValue << ">" << ret;
                        initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    }
                }
                else
                {
                    initmsg( eError ) << "redefinition of local variable  <" << fName << "> with new value <" << fValue << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }
            }
            else
            {
                initmsg( eError ) << "redefinition of external variable  <" << fName << "> with new value <" << fValue << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            }

            fElementState = eElementComplete;
            fName.clear();
            fValue.clear();
            return;
        }

        if( fElementState == eActiveExternalDefine )
        {
            VariableIt GlobalIt = fGlobalMap->find( fName );
            if( GlobalIt == fGlobalMap->end() )
            {
                VariableIt LocalIt = fLocalMap->find( fName );
                if( LocalIt == fLocalMap->end() )
                {
                    VariableIt ExternalIt = fExternalMap->find( fName );
                    if( ExternalIt == fExternalMap->end() )
                    {
                        fExternalMap->insert( VariableEntry( fName, fValue ) );
                    }
                    else
                    {
                        initmsg( eNormal ) << "external variable <" << ExternalIt->first << "> with default value <" << fValue << "> overridden with user value <" << ExternalIt->second << ">" << eom;
                    }
                }
                else
                {
                    initmsg( eError ) << "redefinition of local variable  <" << fName << "> with new value <" << fValue << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }
            }
            else
            {
                initmsg( eError ) << "redefinition of global variable  <" << fName << "> with new value <" << fValue << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            }

            fElementState = eElementComplete;
            fName.clear();
            fValue.clear();
            return;
        }

        if( fElementState == eActiveLocalUndefine )
        {
            VariableIt LocalIt = fLocalMap->find( fName );
            if( LocalIt != fLocalMap->end() )
            {
                fLocalMap->erase( LocalIt );
            }
            else
            {
                VariableIt ExternalIt = fExternalMap->find( fName );
                if( ExternalIt != fExternalMap->end() )
                {
                    initmsg( eError ) << "tried to locally undefine external variable with name <" << fName << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }
                else
                {
                    VariableIt GlobalIt = fGlobalMap->find( fName );
                    if( GlobalIt != fGlobalMap->end() )
                    {
                        initmsg( eError ) << "tried to locally undefine global variable with name <" << fName << ">" << ret;
                        initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    }
                    else
                    {
                        initmsg( eError ) << "variable with name <" << fName << "> is not defined in this file" << ret;
                        initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    }
                }
            }

            fElementState = eElementComplete;
            fName.clear();
            fValue.clear();
            return;
        }

        if( fElementState == eActiveGlobalUndefine )
        {
            VariableIt GlobalIt = fGlobalMap->find( fName );
            if( GlobalIt != fGlobalMap->end() )
            {
                fGlobalMap->erase( GlobalIt );
            }
            else
            {
                VariableIt ExternalIt = fExternalMap->find( fName );
                if( ExternalIt != fExternalMap->end() )
                {
                    initmsg( eError ) << "tried to globally undefine external variable with name <" << fName << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }
                else
                {
                    VariableIt LocalIt = fLocalMap->find( fName );
                    if( LocalIt != fLocalMap->end() )
                    {
                        initmsg( eError ) << "tried to globally undefine local variable with name <" << fName << ">" << ret;
                        initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    }
                    else
                    {
                        initmsg( eError ) << "variable with name <" << fName << "> is not defined" << ret;
                        initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    }
                }
            }

            fElementState = eElementComplete;
            fName.clear();
            fValue.clear();
            return;
        }

        if( fElementState == eActiveExternalUndefine )
        {
            VariableIt ExternalIt = fExternalMap->find( fName );
            if( ExternalIt != fExternalMap->end() )
            {
                fExternalMap->erase( ExternalIt );
            }
            else
            {
                VariableIt GlobalIt = fGlobalMap->find( fName );
                if( GlobalIt != fGlobalMap->end() )
                {
                    initmsg( eError ) << "tried to externally undefine global variable with name <" << fName << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }
                else
                {
                    VariableIt LocalIt = fLocalMap->find( fName );
                    if( LocalIt != fLocalMap->end() )
                    {
                        initmsg( eError ) << "tried to externally undefine local variable with name <" << fName << ">" << ret;
                        initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    }
                    else
                    {
                        initmsg( eError ) << "variable with name <" << fName << "> is not defined" << ret;
                        initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                    }
                }
            }

            fElementState = eElementComplete;
            fName.clear();
            fValue.clear();
            return;
        }

        return;
    }

    void KVariableProcessor::ProcessToken( KElementDataToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            Evaluate( aToken );
            KProcessor::ProcessToken( aToken );
            return;

        }

        if( fElementState == eElementComplete )
        {
            initmsg( eError ) << "got unknown element data <" << aToken->GetValue() << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            return;
        }

        return;
    }

    void KVariableProcessor::ProcessToken( KEndElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementComplete )
        {
            fElementState = eElementInactive;
            return;
        }

        return;
    }

    void KVariableProcessor::ProcessToken( KEndFileToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            delete fLocalMap;
            fLocalMap = fLocalMapStack.top();
            fLocalMapStack.pop();

            KProcessor::ProcessToken( aToken );
            return;
        }

        initmsg( eError ) << "got unknown end of file" << ret;
        initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;

        return;
    }

    void KVariableProcessor::Evaluate( KToken* aToken )
    {
        string tValue;
        string tBuffer;
        stack< string > tBufferStack;

        tValue = aToken->GetValue();

        tBufferStack.push( "" );
        for( size_t Index = 0; Index < tValue.size(); Index++ )
        {
            if( tValue[ Index ] == fStartBracket[ 0 ] )
            {
                tBufferStack.top() += tBuffer;
                tBufferStack.push( "" );
                tBuffer.clear();
                continue;
            }

            if( tValue[ Index ] == fEndBracket[ 0 ] )
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

                string tVarName = tBuffer;
                string tDefaultValue = "";

                size_t tNameValueSepPos = tBuffer.find( fNameValueSeparator[ 0 ] );
                if ( tNameValueSepPos != string::npos )
                {
                    tVarName = tBuffer.substr(0, tNameValueSepPos);
                    tDefaultValue = tBuffer.substr(tNameValueSepPos + 1);
                }

                VariableIt ExternalVariable = fExternalMap->find( tVarName );
                if( ExternalVariable != fExternalMap->end() )
                {
                    tBuffer = ExternalVariable->second;
                }
                else
                {
                    VariableIt GlobalVariable = fGlobalMap->find( tVarName );
                    if( GlobalVariable != fGlobalMap->end() )
                    {
                        tBuffer = GlobalVariable->second;
                    }
                    else
                    {
                        VariableIt FileVariable = fLocalMap->find( tVarName );
                        if( FileVariable != fLocalMap->end() )
                        {
                            tBuffer = FileVariable->second;
                        }
                        else if ( tNameValueSepPos != string::npos )
                        {
                            tBuffer = tDefaultValue;
                        }
                        else
                        {
                            initmsg( eError ) << "variable <" << tBuffer << ">" << " is not defined" << ret;
                            initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                        }
                    }
                }

                tBufferStack.top() += tBuffer;
                tBuffer.clear();
                continue;
            }

            tBuffer.append( 1, tValue[ Index ] );
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

}
