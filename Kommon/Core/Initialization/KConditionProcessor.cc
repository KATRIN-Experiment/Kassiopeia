#include "KConditionProcessor.hh"
#include "KInitializationMessage.hh"

#include <iostream>

using namespace std;

namespace katrin
{

    KConditionProcessor::KConditionProcessor() :
        KProcessor(),
        fElementState( eElementInactive ),
        fAttributeState( eAttributeInactive ),
        fNest( 0 ),
        fCondition( false ),
        fIfTokens(),
        fNewParent( NULL ),
        fOldParent( NULL )
    {

    }

    KConditionProcessor::~KConditionProcessor()
    {
    }

    void KConditionProcessor::ProcessToken( KBeginElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            if( aToken->GetValue() == string( "if" ) )
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

        if( fElementState == eElementComplete )
        {

            if( aToken->GetValue() == string( "if" ) )
            {
                fNest++;
            }
            fIfTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KConditionProcessor::ProcessToken( KBeginAttributeToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            if( aToken->GetValue() == "condition" )
            {
                fAttributeState = eCondition;
                return;
            }

            initmsg( eError ) << "got unknown attribute <" << aToken->GetValue() << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;

            return;
        }


        if( fElementState == eElementComplete )
        {
        	fIfTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KConditionProcessor::ProcessToken( KAttributeDataToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            if( fAttributeState == eCondition )
            {
                const string condStr = aToken->GetValue();
                if( condStr.find_first_of("{}[]") != string::npos )
                {
                    initmsg(eError) << "A condition containing an unevaluated "
                        << "formula {} or variable [] could not be interpreted." << eom;
                    fCondition = false;
                }
                else
                {
                    if( aToken->GetValue< string >().empty() )
                    {
                        fCondition = false;
                    }
                    else
                    {
                        fCondition = aToken->GetValue< bool >();
                    }
                }
                fAttributeState = eAttributeComplete;
                return;
            }
        }

        if( fElementState == eElementComplete )
        {
        	fIfTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KConditionProcessor::ProcessToken( KEndAttributeToken* aToken )
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
        	fIfTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KConditionProcessor::ProcessToken( KMidElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
        	//hijack the token stream
            Remove();
            InsertAfter( fNewParent );
            fElementState = eElementComplete;
            return;
        }

        if( fElementState == eElementComplete )
        {
        	fIfTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KConditionProcessor::ProcessToken( KElementDataToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementComplete )
        {
        	fIfTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KConditionProcessor::ProcessToken( KEndElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementComplete )
        {
            //see if the end element is a "if"; if so, decrease the loop counter
            if( aToken->GetValue() == string( "if" ) )
            {
                fNest--;

                //if we're at the end of the loop counter
                if( fNest == 0 )
                {
                    //un-hijack the tokenizer
                    Remove();
                    InsertAfter( fOldParent );


                    //copy the state into stack variables and reset (otherwise nesting is not possible)
                    TokenVector tIfTokens = fIfTokens;
                    bool tCondition = fCondition;

                    //reset the object
                    fNest = 0;
                    fCondition = false;
                    fIfTokens.clear();
                    fNewParent = NULL;
                    fOldParent = NULL;
                    fElementState = eElementInactive;

                    //utilities
                    KToken* tToken;

                    if ( tCondition == true )
                    {
						for( TokenIt It = tIfTokens.begin(); It != tIfTokens.end(); It++ )
						{
							tToken = (*It)->Clone();
							Dispatch( tToken );
							delete tToken;
						}
                    }

					//delete the old tokens (made with new during collection)
					for( TokenIt It = tIfTokens.begin(); It != tIfTokens.end(); It++ )
					{
						delete *It;
					}
                    return;
                }
            }

            fIfTokens.push_back( aToken->Clone() );
            return;
        }

        return;
    }

    void KConditionProcessor::Dispatch( KToken* aToken )
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
