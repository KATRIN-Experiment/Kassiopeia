#include "KConditionProcessor.hh"

#include "KInitializationMessage.hh"

#include <iostream>

using namespace std;

namespace katrin
{

KConditionProcessor::KConditionProcessor() :
    KProcessor(),
    fElementState(eElementInactive),
    fAttributeState(eAttributeInactive),
    fProcessorState(eIfCondition),
    fNest(0),
    fCondition(false),
    fIfTokens(),
    fElseTokens(),
    fNewParent(nullptr),
    fOldParent(nullptr)
{}

KConditionProcessor::~KConditionProcessor() = default;

void KConditionProcessor::ProcessToken(KBeginElementToken* aToken)
{
    if (fElementState == eElementInactive) {
        if (aToken->GetValue() == string("if")) {
            fNest++;

            if (fNest == 1) {
                fOldParent = this->fParent;
                fNewParent = this->GetFirstParent();
                fElementState = eActive;
                fProcessorState = eIfCondition;
                return;
            }
        }

        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementComplete) {
        if (aToken->GetValue() == string("else")) {
            fNest++;

            if (fNest == 2) {
                //fOldParent = this->fParent;
                //fNewParent = this->GetFirstParent();
                fElementState = eElseActive;
                fProcessorState = eElseCondition;
                return;
            }
        }

        if (aToken->GetValue() == string("if")) {
            fNest++;
        }

        if (fProcessorState == eIfCondition) {
            fIfTokens.push_back(aToken->Clone());
        }
        else if (fProcessorState == eElseCondition) {
            fElseTokens.push_back(aToken->Clone());
        }
        return;
    }

    return;
}

void KConditionProcessor::ProcessToken(KBeginAttributeToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActive) {
        if (aToken->GetValue() == "condition") {
            fAttributeState = eCondition;
            return;
        }

        initmsg(eError) << "got unknown attribute <" << aToken->GetValue() << ">" << ret;
        initmsg(eError) << "in path <" << aToken->GetPath() << "in file <" << aToken->GetFile() << "> at line <"
                        << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;

        return;
    }

    if (fElementState == eElementComplete) {
        if (fProcessorState == eIfCondition) {
            fIfTokens.push_back(aToken->Clone());
        }
        else if (fProcessorState == eElseCondition) {
            fElseTokens.push_back(aToken->Clone());
        }
        return;
    }

    return;
}

void KConditionProcessor::ProcessToken(KAttributeDataToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActive) {
        if (fAttributeState == eCondition) {
            const string condStr = aToken->GetValue();
            if (condStr.find_first_of("{}[]") != string::npos) {
                initmsg(eError) << "A condition containing an unevaluated "
                                << "formula {} or variable [] could not be interpreted:\n"
                                << condStr << eom;
                fCondition = false;
            }
            else {
                if (aToken->GetValue<string>().empty()) {
                    fCondition = false;
                }
                else {
                    fCondition = aToken->GetValue<bool>();
                }
            }
            fAttributeState = eAttributeComplete;
            return;
        }
    }

    if (fElementState == eElementComplete) {
        if (fProcessorState == eIfCondition) {
            fIfTokens.push_back(aToken->Clone());
        }
        else if (fProcessorState == eElseCondition) {
            fElseTokens.push_back(aToken->Clone());
        }
        return;
    }

    return;
}

void KConditionProcessor::ProcessToken(KEndAttributeToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActive) {
        if (fAttributeState == eAttributeComplete) {
            fAttributeState = eAttributeInactive;
            return;
        }
    }

    if (fElementState == eElementComplete) {
        if (fProcessorState == eIfCondition) {
            fIfTokens.push_back(aToken->Clone());
        }
        else if (fProcessorState == eElseCondition) {
            fElseTokens.push_back(aToken->Clone());
        }
        return;
    }

    return;
}

void KConditionProcessor::ProcessToken(KMidElementToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActive || fElementState == eElseActive) {
        //hijack the token stream
        Remove();
        InsertAfter(fNewParent);
        fElementState = eElementComplete;
        return;
    }

    if (fElementState == eElementComplete) {
        if (fProcessorState == eIfCondition) {
            fIfTokens.push_back(aToken->Clone());
        }
        else if (fProcessorState == eElseCondition) {
            fElseTokens.push_back(aToken->Clone());
        }
        return;
    }

    return;
}

void KConditionProcessor::ProcessToken(KElementDataToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementComplete) {
        if (fProcessorState == eIfCondition) {
            fIfTokens.push_back(aToken->Clone());
        }
        else if (fProcessorState == eElseCondition) {
            fElseTokens.push_back(aToken->Clone());
        }
        return;
    }

    return;
}

void KConditionProcessor::ProcessToken(KEndElementToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementComplete) {
        //see if the end element is a "if"; if so, decrease the loop counter
        if (aToken->GetValue() == string("if")) {
            fNest--;

            //if we're at the end of the loop counter
            if (fNest == 0) {
                //un-hijack the tokenizer
                Remove();
                InsertAfter(fOldParent);

                //copy the state into stack variables and reset (otherwise nesting is not possible)
                TokenVector tIfTokens = fIfTokens;
                TokenVector tElseTokens = fElseTokens;
                bool tCondition = fCondition;

                //reset the object
                fNest = 0;
                fCondition = false;
                fIfTokens.clear();
                fElseTokens.clear();
                fNewParent = nullptr;
                fOldParent = nullptr;
                fElementState = eElementInactive;

                //utilities
                KToken* tToken;

                if (tCondition == true) {
                    for (auto& ifToken : tIfTokens) {
                        tToken = ifToken->Clone();
                        Dispatch(tToken);
                        delete tToken;
                    }
                }
                else {
                    for (auto& elseToken : tElseTokens) {
                        tToken = elseToken->Clone();
                        Dispatch(tToken);
                        delete tToken;
                    }
                }

                //delete the old tokens (made with new during collection)
                for (auto& ifToken : tIfTokens) {
                    delete ifToken;
                }
                for (auto& elseToken : tElseTokens) {
                    delete elseToken;
                }
                return;
            }
        }

        if (aToken->GetValue() == string("else")) {
            fNest--;
        }

        if (fProcessorState == eIfCondition) {
            fIfTokens.push_back(aToken->Clone());
        }
        else if (fProcessorState == eElseCondition) {
            fElseTokens.push_back(aToken->Clone());
        }
        return;
    }

    return;
}

void KConditionProcessor::Dispatch(KToken* aToken)
{
    KBeginParsingToken* tBeginParsingToken = nullptr;
    tBeginParsingToken = dynamic_cast<KBeginParsingToken*>(aToken);
    if (tBeginParsingToken != nullptr) {
        GetFirstParent()->ProcessToken(tBeginParsingToken);
        return;
    }

    KBeginFileToken* tBeginFileToken = nullptr;
    tBeginFileToken = dynamic_cast<KBeginFileToken*>(aToken);
    if (tBeginFileToken != nullptr) {
        GetFirstParent()->ProcessToken(tBeginFileToken);
        return;
    }

    KBeginElementToken* tBeginElementToken = nullptr;
    tBeginElementToken = dynamic_cast<KBeginElementToken*>(aToken);
    if (tBeginElementToken != nullptr) {
        GetFirstParent()->ProcessToken(tBeginElementToken);
        return;
    }

    KBeginAttributeToken* tBeginAttributeToken = nullptr;
    tBeginAttributeToken = dynamic_cast<KBeginAttributeToken*>(aToken);
    if (tBeginAttributeToken != nullptr) {
        GetFirstParent()->ProcessToken(tBeginAttributeToken);
        return;
    }

    KAttributeDataToken* tAttributeDataToken = nullptr;
    tAttributeDataToken = dynamic_cast<KAttributeDataToken*>(aToken);
    if (tAttributeDataToken != nullptr) {
        GetFirstParent()->ProcessToken(tAttributeDataToken);
        return;
    }

    KEndAttributeToken* tEndAttributeToken = nullptr;
    tEndAttributeToken = dynamic_cast<KEndAttributeToken*>(aToken);
    if (tEndAttributeToken != nullptr) {
        GetFirstParent()->ProcessToken(tEndAttributeToken);
        return;
    }

    KMidElementToken* tMidElementToken = nullptr;
    tMidElementToken = dynamic_cast<KMidElementToken*>(aToken);
    if (tMidElementToken != nullptr) {
        GetFirstParent()->ProcessToken(tMidElementToken);
        return;
    }

    KElementDataToken* tElementDataToken = nullptr;
    tElementDataToken = dynamic_cast<KElementDataToken*>(aToken);
    if (tElementDataToken != nullptr) {
        GetFirstParent()->ProcessToken(tElementDataToken);
        return;
    }

    KEndElementToken* tEndElementToken = nullptr;
    tEndElementToken = dynamic_cast<KEndElementToken*>(aToken);
    if (tEndElementToken != nullptr) {
        GetFirstParent()->ProcessToken(tEndElementToken);
        return;
    }

    KEndFileToken* tEndFileToken = nullptr;
    tEndFileToken = dynamic_cast<KEndFileToken*>(aToken);
    if (tEndFileToken != nullptr) {
        GetFirstParent()->ProcessToken(tEndFileToken);
        return;
    }

    KEndParsingToken* tEndParsingToken = nullptr;
    tEndParsingToken = dynamic_cast<KEndParsingToken*>(aToken);
    if (tEndParsingToken != nullptr) {
        GetFirstParent()->ProcessToken(tEndParsingToken);
        return;
    }

    KCommentToken* tCommentToken = nullptr;
    tCommentToken = dynamic_cast<KCommentToken*>(aToken);
    if (tCommentToken != nullptr) {
        GetFirstParent()->ProcessToken(tCommentToken);
        return;
    }

    KErrorToken* tErrorToken = nullptr;
    tErrorToken = dynamic_cast<KErrorToken*>(aToken);
    if (tErrorToken != nullptr) {
        GetFirstParent()->ProcessToken(tErrorToken);
        return;
    }

    return;
}

}  // namespace katrin
