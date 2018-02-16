#include "KVariableProcessor.hh"
#include "KInitializationMessage.hh"
#include "KToolbox.h"

using namespace std;

namespace katrin
{
    const string KVariableProcessor::fStartBracket = "[";
    const string KVariableProcessor::fEndBracket = "]";
    const string KVariableProcessor::fNameValueSeparator = ":";
    const string KVariableProcessor::fAppendValueSeparator = " ";

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
        for ( auto & tRefCount : fRefCountMap )
        {
            if ( tRefCount.second > 0 )
            {
                initmsg_debug( "variable <" << tRefCount.first << "> was referenced " << tRefCount.second << " times" << eom );
            }
            else
            {
                initmsg( eInfo ) << "variable <" << tRefCount.first << "> was not referenced anywhere" << eom;
            }
        }

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
            initmsg_debug( "variable processor found token <" << aToken->GetValue() << ">" << eom)

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

            if( aToken->GetValue() == string( "redefine" ) )
            {
                fElementState = eActiveLocalRedefine;
                return;
            }

            if( aToken->GetValue() == string( "global_redefine" ) )
            {
                fElementState = eActiveGlobalRedefine;
                return;
            }

            if( aToken->GetValue() == string( "external_redefine" ) )
            {
                fElementState = eActiveExternalRedefine;
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

            if( aToken->GetValue() == string( "append" ) )
            {
                fElementState = eActiveLocalAppend;
                return;
            }

            if( aToken->GetValue() == string( "global_append" ) )
            {
                fElementState = eActiveGlobalAppend;
                return;
            }

            if( aToken->GetValue() == string( "external_append" ) )
            {
                fElementState = eActiveExternalAppend;
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
        if( (fElementState == eActiveLocalDefine) || (fElementState == eActiveGlobalDefine) || (fElementState == eActiveExternalDefine) || (fElementState == eActiveLocalRedefine) || (fElementState == eActiveGlobalRedefine) || (fElementState == eActiveExternalRedefine) || (fElementState == eActiveLocalAppend) || (fElementState == eActiveGlobalAppend) || (fElementState == eActiveExternalAppend) )
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

        if( (fElementState == eActiveLocalDefine) || (fElementState == eActiveGlobalDefine) || (fElementState == eActiveExternalDefine) || (fElementState == eActiveLocalRedefine) || (fElementState == eActiveGlobalRedefine) || (fElementState == eActiveExternalRedefine) || (fElementState == eActiveLocalUndefine) || (fElementState == eActiveGlobalUndefine) || (fElementState == eActiveExternalUndefine) || (fElementState == eActiveLocalAppend) || (fElementState == eActiveGlobalAppend) || (fElementState == eActiveExternalAppend) )
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

        if( (fElementState == eActiveLocalDefine) || (fElementState == eActiveGlobalDefine) || (fElementState == eActiveExternalDefine) || (fElementState == eActiveLocalRedefine) || (fElementState == eActiveGlobalRedefine) || (fElementState == eActiveExternalRedefine) || (fElementState == eActiveLocalUndefine) || (fElementState == eActiveGlobalUndefine) || (fElementState == eActiveExternalUndefine) || (fElementState == eActiveLocalAppend) || (fElementState == eActiveGlobalAppend) || (fElementState == eActiveExternalAppend) )
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

        KToolbox& toolbox = KToolbox::GetInstance();

        // define

        if( fElementState == eActiveLocalDefine )
        {
            if( toolbox.Get(fName) == nullptr )
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
                            initmsg_debug( "creation of local variable  <" << fName << "> with new value <" << fValue << ">" << eom );
                            fLocalMap->insert( VariableEntry( fName, fValue ) );
                            //fRefCountMap[fName] = 0;
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
            }
            else
            {
                initmsg( eError ) << "redefinition of toolbox variable  <" << fName << "> with new value <" << fValue << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            }

            fElementState = eElementComplete;
            fName.clear();
            fValue.clear();
            return;
        }

        if( fElementState == eActiveGlobalDefine )
        {
            if( toolbox.Get(fName) == nullptr )
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
                            initmsg_debug( "creation of global variable  <" << fName << "> with new value <" << fValue << ">" << eom );
                            fGlobalMap->insert( VariableEntry( fName, fValue ) );
                            fRefCountMap[fName] = 0;
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
            }
            else
            {
                initmsg( eError ) << "redefinition of toolbox variable  <" << fName << "> with new value <" << fValue << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            }

            fElementState = eElementComplete;
            fName.clear();
            fValue.clear();
            return;
        }

        if( fElementState == eActiveExternalDefine )
        {
            if( toolbox.Get(fName) == nullptr )
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
                            initmsg_debug( "creation of external variable  <" << fName << "> with new value <" << fValue << ">" << eom );
                            fExternalMap->insert( VariableEntry( fName, fValue ) );
                            fRefCountMap[fName] = 0;
                        }
                        else
                        {
                            initmsg( eInfo ) << "external variable <" << ExternalIt->first << "> with default value <" << fValue << "> overridden with user value <" << ExternalIt->second << ">" << eom;
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
            }
            else
            {
                initmsg( eError ) << "redefinition of toolbox variable  <" << fName << "> with new value <" << fValue << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            }

            fElementState = eElementComplete;
            fName.clear();
            fValue.clear();
            return;
        }

        // redefine

        if( fElementState == eActiveLocalRedefine )
        {
            VariableIt LocalIt = fLocalMap->find( fName );
            if( LocalIt != fLocalMap->end() )
            {
                initmsg_debug( "redefinition of local variable  <" << fName << "> with new value <" << fValue << ">, old value was <" << LocalIt->second << ">" << eom );
                LocalIt->second = fValue;
            }
            else
            {
                VariableIt ExternalIt = fExternalMap->find( fName );
                if( ExternalIt != fExternalMap->end() )
                {
                    initmsg( eError ) << "tried to locally redefine external variable with name <" << fName << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }
                else
                {
                    VariableIt GlobalIt = fGlobalMap->find( fName );
                    if( GlobalIt != fGlobalMap->end() )
                    {
                        initmsg( eError ) << "tried to locally redefine global variable with name <" << fName << ">" << ret;
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

        if( fElementState == eActiveGlobalRedefine )
        {
            VariableIt GlobalIt = fGlobalMap->find( fName );
            if( GlobalIt != fGlobalMap->end() )
            {
                initmsg_debug( "redefinition of global variable  <" << fName << "> with new value <" << fValue << ">, old value was <" << GlobalIt->second << ">" << eom );
                GlobalIt->second = fValue;
            }
            else
            {
                VariableIt ExternalIt = fExternalMap->find( fName );
                if( ExternalIt != fExternalMap->end() )
                {
                    initmsg( eError ) << "tried to globally redefine external variable with name <" << fName << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }
                else
                {
                    VariableIt LocalIt = fLocalMap->find( fName );
                    if( LocalIt != fLocalMap->end() )
                    {
                        initmsg( eError ) << "tried to globally redefine local variable with name <" << fName << ">" << ret;
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

        if( fElementState == eActiveExternalRedefine )
        {
            VariableIt ExternalIt = fExternalMap->find( fName );
            if( ExternalIt != fExternalMap->end() )
            {
                initmsg_debug( "redefinition of external variable  <" << fName << "> with new value <" << fValue << ">, old value was <" << ExternalIt->second << ">" << eom );
                ExternalIt->second = fValue;
            }
            else
            {
                VariableIt GlobalIt = fGlobalMap->find( fName );
                if( GlobalIt != fGlobalMap->end() )
                {
                    initmsg( eError ) << "tried to externally redefine global variable with name <" << fName << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }
                else
                {
                    VariableIt LocalIt = fLocalMap->find( fName );
                    if( LocalIt != fLocalMap->end() )
                    {
                        initmsg( eError ) << "tried to externally redefine local variable with name <" << fName << ">" << ret;
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

        // undefine

        if( fElementState == eActiveLocalUndefine )
        {
            VariableIt LocalIt = fLocalMap->find( fName );
            if( LocalIt != fLocalMap->end() )
            {
                initmsg_debug( "deletion of local variable  <" << fName << "> with current value <" << LocalIt->second << ">" << eom );
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
                initmsg_debug( "deletion of global variable  <" << fName << "> with current value <" << GlobalIt->second << ">" << eom );
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
                initmsg_debug( "deletion of external variable  <" << fName << "> with current value <" << ExternalIt->second << ">" << eom );
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

        // append

        if( fElementState == eActiveLocalAppend )
        {
            VariableIt LocalIt = fLocalMap->find( fName );
            if( LocalIt != fLocalMap->end() )
            {
                initmsg_debug( "redefinition of local variable  <" << fName << "> by appending value <" << fValue << ">, old value was <" << LocalIt->second << ">" << eom );
                LocalIt->second += string(LocalIt->second.empty() ? "" : fAppendValueSeparator) + fValue;
            }
            else
            {
                VariableIt ExternalIt = fExternalMap->find( fName );
                if( ExternalIt != fExternalMap->end() )
                {
                    initmsg( eError ) << "tried to locally append to external variable with name <" << fName << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }
                else
                {
                    VariableIt GlobalIt = fGlobalMap->find( fName );
                    if( GlobalIt != fGlobalMap->end() )
                    {
                        initmsg( eError ) << "tried to locally append to global variable with name <" << fName << ">" << ret;
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

        if( fElementState == eActiveGlobalAppend )
        {
            VariableIt GlobalIt = fGlobalMap->find( fName );
            if( GlobalIt != fGlobalMap->end() )
            {
                initmsg_debug( "redefinition of global variable  <" << fName << "> by appending value <" << fValue << ">, old value was <" << GlobalIt->second << ">" << eom );
                GlobalIt->second += string(GlobalIt->second.empty() ? "" : fAppendValueSeparator) + fValue;
            }
            else
            {
                VariableIt ExternalIt = fExternalMap->find( fName );
                if( ExternalIt != fExternalMap->end() )
                {
                    initmsg( eError ) << "tried to globally append to external variable with name <" << fName << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }
                else
                {
                    VariableIt LocalIt = fLocalMap->find( fName );
                    if( LocalIt != fLocalMap->end() )
                    {
                        initmsg( eError ) << "tried to globally append to local variable with name <" << fName << ">" << ret;
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

        if( fElementState == eActiveExternalAppend )
        {
            VariableIt ExternalIt = fExternalMap->find( fName );
            if( ExternalIt != fExternalMap->end() )
            {
                initmsg_debug( "redefinition of external variable  <" << fName << "> by appending value <" << fValue << ">, old value was <" << ExternalIt->second << ">" << eom );
                ExternalIt->second += string(ExternalIt->second.empty() ? "" : fAppendValueSeparator) + fValue;
            }
            else
            {
                VariableIt GlobalIt = fGlobalMap->find( fName );
                if( GlobalIt != fGlobalMap->end() )
                {
                    initmsg( eError ) << "tried to externally append to global variable with name <" << fName << ">" << ret;
                    initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                }
                else
                {
                    VariableIt LocalIt = fLocalMap->find( fName );
                    if( LocalIt != fLocalMap->end() )
                    {
                        initmsg( eError ) << "tried to externally append to local variable with name <" << fName << ">" << ret;
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

        KToolbox& toolbox = KToolbox::GetInstance();

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

                const string* tToolboxVariable = toolbox.Get<string>(tVarName);
                if( tToolboxVariable != nullptr )
                {
                    tBuffer = *tToolboxVariable;
                }
                else
                {
                    VariableIt ExternalVariable = fExternalMap->find( tVarName );
                    if( ExternalVariable != fExternalMap->end() )
                    {
                        fRefCountMap[tVarName]++;
                        tBuffer = ExternalVariable->second;
                    }
                    else
                    {
                        VariableIt GlobalVariable = fGlobalMap->find( tVarName );
                        if( GlobalVariable != fGlobalMap->end() )
                        {
                            fRefCountMap[tVarName]++;
                            tBuffer = GlobalVariable->second;
                        }
                        else
                        {
                            VariableIt LocalVariable = fLocalMap->find( tVarName );
                            if( LocalVariable != fLocalMap->end() )
                            {
                                //fRefCountMap[tVarName]++;
                                tBuffer = LocalVariable->second;
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
