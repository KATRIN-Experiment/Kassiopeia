#include "KLoopProcessor.hh"
#include "KInitializationMessage.hh"

#include <iostream>
using std::cout;
using std::endl;

#include <cstdlib>

namespace katrin
{

    const string KLoopProcessor::fStartBracket = "[";
    const string KLoopProcessor::fEndBracket = "]";

    KLoopProcessor::KLoopProcessor() :
        KProcessor(),
        fElementState( eElementInactive ),
        fAttributeState( eAttributeInactive ),
        fNest( 0 ),
        fVariable( "" ),
        fStartValue( 0 ),
        fEndValue( 0 ),
        fStepValue( 0 ),
        fTokens(),
        fNewParent( NULL ),
        fOldParent( NULL )
    {

    }

    KLoopProcessor::~KLoopProcessor()
    {
    }

    void KLoopProcessor::ProcessToken( KBeginElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            if( aToken->GetValue() == string( "loop" ) )
            {
                fNest++;

                if( fNest == 1 )
                {
                    fOldParent = this->fParent;
                    fNewParent = this->GetFirstParent();
                    fElementState = eActive;
                    return;
                }
            }

            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            Remove();
            InsertAfter( fNewParent );

            fElementState = eElementComplete;
            ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementComplete )
        {
            if( aToken->GetValue() == string( "loop" ) )
            {
                fNest++;
            }
            fTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KLoopProcessor::ProcessToken( KBeginAttributeToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            if( aToken->GetValue() == "variable" )
            {
                fAttributeState = eVariable;
                return;
            }

            if( aToken->GetValue() == "start" )
            {
                fAttributeState = eStart;
                return;
            }

            if( aToken->GetValue() == "end" )
            {
                fAttributeState = eEnd;
                return;
            }

            if( aToken->GetValue() == "step" )
            {
                fAttributeState = eStep;
                return;
            }

            initmsg( eError ) << "got unknown attribute <" << aToken->GetValue() << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;

            return;
        }

        if( fElementState == eElementComplete )
        {
            fTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KLoopProcessor::ProcessToken( KAttributeDataToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            if( fAttributeState == eVariable )
            {
                fVariable = aToken->GetValue();
                fAttributeState = eAttributeComplete;
                return;
            }

            if( fAttributeState == eStart )
            {
                fStartValue = aToken->GetValue< int >();
                fAttributeState = eAttributeComplete;
                return;
            }

            if( fAttributeState == eEnd )
            {
                fEndValue = aToken->GetValue< int >();
                fAttributeState = eAttributeComplete;
                return;
            }

            if( fAttributeState == eStep )
            {
                fStepValue = aToken->GetValue< int >();
                fAttributeState = eAttributeComplete;
                return;
            }
        }

        if( fElementState == eElementComplete )
        {
            fTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KLoopProcessor::ProcessToken( KEndAttributeToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            if( fAttributeState == eAttributeComplete )
            {
                fAttributeState = eAttributeInactive;
                return;
            }
        }

        if( fElementState == eElementComplete )
        {
            fTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KLoopProcessor::ProcessToken( KMidElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            Remove();
            InsertAfter( fNewParent );

            fElementState = eElementComplete;

            return;
        }

        if( fElementState == eElementComplete )
        {
            fTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KLoopProcessor::ProcessToken( KElementDataToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementComplete )
        {
            fTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KLoopProcessor::ProcessToken( KEndElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementComplete )
        {
            //see if the end element is a loop; if so, decrease the loop counter
            if( aToken->GetValue() == string( "loop" ) )
            {
                fNest--;

                //if we're at the end of the loop counter
                if( fNest == 0 )
                {
                    //un-hijack the tokenizer
                    Remove();
                    InsertAfter( fOldParent );

                    //copy the state into stack variables and reset (otherwise nesting is not possible)
                    string tVariable = fVariable;
                    int tStart = fStartValue;
                    int tEnd = fEndValue;
                    int tStep = fStepValue;
                    TokenVector tTokens = fTokens;

                    //reset the object
                    fElementState = eElementInactive;
                    fNest = 0;
                    fVariable = "";
                    fStartValue = 0;
                    fEndValue = 0;
                    fStepValue = 0;
                    fTokens.clear();
                    fNewParent = NULL;
                    fOldParent = NULL;

                    //utilities
                    int tIndex;
                    stringstream tConverter;
                    KToken* tToken;

                    //run the loop
                    tIndex = tStart;
                    while( true )
                    {
                        tConverter.clear();
                        tConverter.str( "" );
                        tConverter << tIndex;

                        for( TokenIt It = tTokens.begin(); It != tTokens.end(); It++ )
                        {
                            tToken = (*It)->Clone();
                            Evaluate( tToken, tVariable, tConverter.str() );
                            Dispatch( tToken );
                            delete tToken;
                        }

                        tIndex += tStep;

                        if( (tStep < 0) && (tIndex < tEnd) )
                        {
                            break;
                        }

                        if( (tStep > 0) && (tIndex > tEnd) )
                        {
                            break;
                        }
                    }

                    //delete the old tokens (made with new during collection)
                    for( TokenIt It = tTokens.begin(); It != tTokens.end(); It++ )
                    {
                        delete *It;
                    }

                    return;
                }
            }

            fTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KLoopProcessor::Reset()
    {

        return;
    }
    void KLoopProcessor::Evaluate( KToken* aToken, const string& aName, const string& aValue )
    {
        string tValue;
        string tBuffer;
        stack< string > tBufferStack;

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

                if( tBuffer == aName )
                {
                    tBuffer = aValue;
                }
                else
                {
                    tBuffer = fStartBracket + tBuffer + fEndBracket;
                }

                tBufferStack.top() += tBuffer;
                tBuffer.clear();
                continue;
            }

            tBuffer.append( 1, tValue[Index] );
        }

        tBufferStack.top() += tBuffer;
        tBuffer = tBufferStack.top();
        tBufferStack.pop();
        if( tBufferStack.size() != 0 )
        {
            initmsg( eError ) << "bracket matching problem at end of string <" << tValue << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            return;
        }
        tValue = tBuffer;

        aToken->SetValue( tValue );

        return;
    }
    void KLoopProcessor::Dispatch( KToken* aToken )
    {
        KBeginParsingToken* tBeginParsingToken = NULL;
        tBeginParsingToken = dynamic_cast< KBeginParsingToken* >( aToken );
        if( tBeginParsingToken != NULL )
        {
            GetFirstParent()->ProcessToken( tBeginParsingToken );
            return;
        }

        KBeginFileToken* tBeginFileToken = NULL;
        tBeginFileToken = dynamic_cast< KBeginFileToken* >( aToken );
        if( tBeginFileToken != NULL )
        {
            GetFirstParent()->ProcessToken( tBeginFileToken );
            return;
        }

        KBeginElementToken* tBeginElementToken = NULL;
        tBeginElementToken = dynamic_cast< KBeginElementToken* >( aToken );
        if( tBeginElementToken != NULL )
        {
            GetFirstParent()->ProcessToken( tBeginElementToken );
            return;
        }

        KBeginAttributeToken* tBeginAttributeToken = NULL;
        tBeginAttributeToken = dynamic_cast< KBeginAttributeToken* >( aToken );
        if( tBeginAttributeToken != NULL )
        {
            GetFirstParent()->ProcessToken( tBeginAttributeToken );
            return;
        }

        KAttributeDataToken* tAttributeDataToken = NULL;
        tAttributeDataToken = dynamic_cast< KAttributeDataToken* >( aToken );
        if( tAttributeDataToken != NULL )
        {
            GetFirstParent()->ProcessToken( tAttributeDataToken );
            return;
        }

        KEndAttributeToken* tEndAttributeToken = NULL;
        tEndAttributeToken = dynamic_cast< KEndAttributeToken* >( aToken );
        if( tEndAttributeToken != NULL )
        {
            GetFirstParent()->ProcessToken( tEndAttributeToken );
            return;
        }

        KMidElementToken* tMidElementToken = NULL;
        tMidElementToken = dynamic_cast< KMidElementToken* >( aToken );
        if( tMidElementToken != NULL )
        {
            GetFirstParent()->ProcessToken( tMidElementToken );
            return;
        }

        KElementDataToken* tElementDataToken = NULL;
        tElementDataToken = dynamic_cast< KElementDataToken* >( aToken );
        if( tElementDataToken != NULL )
        {
            GetFirstParent()->ProcessToken( tElementDataToken );
            return;
        }

        KEndElementToken* tEndElementToken = NULL;
        tEndElementToken = dynamic_cast< KEndElementToken* >( aToken );
        if( tEndElementToken != NULL )
        {
            GetFirstParent()->ProcessToken( tEndElementToken );
            return;
        }

        KEndFileToken* tEndFileToken = NULL;
        tEndFileToken = dynamic_cast< KEndFileToken* >( aToken );
        if( tEndFileToken != NULL )
        {
            GetFirstParent()->ProcessToken( tEndFileToken );
            return;
        }

        KEndParsingToken* tEndParsingToken = NULL;
        tEndParsingToken = dynamic_cast< KEndParsingToken* >( aToken );
        if( tEndParsingToken != NULL )
        {
            GetFirstParent()->ProcessToken( tEndParsingToken );
            return;
        }

        KCommentToken* tCommentToken = NULL;
        tCommentToken = dynamic_cast< KCommentToken* >( aToken );
        if( tCommentToken != NULL )
        {
            GetFirstParent()->ProcessToken( tCommentToken );
            return;
        }

        KErrorToken* tErrorToken = NULL;
        tErrorToken = dynamic_cast< KErrorToken* >( aToken );
        if( tErrorToken != NULL )
        {
            GetFirstParent()->ProcessToken( tErrorToken );
            return;
        }

        return;
    }

}
