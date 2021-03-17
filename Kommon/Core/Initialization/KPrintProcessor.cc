#include "KPrintProcessor.hh"

#include "KInitializationMessage.hh"

#include <iostream>

using namespace std;

namespace katrin
{

KPrintProcessor::KPrintProcessor() :
    KProcessor(),
    fElementState(eElementInactive),
    fAttributeState(eAttributeInactive),
    fMessageType(eNormal),
    fCheckAssertCondition(false),
    fName(""),
    fValue(""),
    fAssertCondition(false)
{}

KPrintProcessor::~KPrintProcessor() = default;

void KPrintProcessor::ProcessToken(KBeginElementToken* aToken)
{
    fCheckAssertCondition = false;
    if (fElementState == eElementInactive) {
        if (aToken->GetValue() == string("print")) {
            fElementState = eElementActive;
            fMessageType = eNormal;
            return;
        }
        if (aToken->GetValue() == string("warning")) {
            fElementState = eElementActive;
            fMessageType = eWarning;
            return;
        }
        if (aToken->GetValue() == string("error")) {
            fElementState = eElementActive;
            fMessageType = eError;
            return;
        }
        if (aToken->GetValue() == string("assert")) {
            fCheckAssertCondition = true;
            fElementState = eElementActive;
            fMessageType = eDebug;
            return;
        }

        KProcessor::ProcessToken(aToken);
        return;
    }
    return;
}

void KPrintProcessor::ProcessToken(KBeginAttributeToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementActive) {
        if (aToken->GetValue() == "name") {
            fAttributeState = eActiveName;
            return;
        }
        if (!fCheckAssertCondition) {
            if (aToken->GetValue() == "value") {
                fAttributeState = eActiveValue;
                return;
            }
        }
        else {
            if (aToken->GetValue() == "condition") {
                fAttributeState = eActiveAssertCondition;
                return;
            }
        }

        initmsg(eError) << "got unknown attribute <" << aToken->GetValue() << ">" << ret;
        initmsg(eError) << "in path <" << aToken->GetPath() << "in file <" << aToken->GetFile() << "> at line <"
                        << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;

        return;
    }
    return;
}

void KPrintProcessor::ProcessToken(KAttributeDataToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementActive) {
        if (fAttributeState == eActiveName) {
            fName = aToken->GetValue<string>();
            fAttributeState = eAttributeComplete;
            return;
        }
        if (fAttributeState == eActiveValue) {
            fValue = aToken->GetValue<string>();
            fAttributeState = eAttributeComplete;
            return;
        }
        if (fAttributeState == eActiveAssertCondition) {
            fAssertCondition = false;
            const string condStr = aToken->GetValue();
            if (condStr.find_first_of("{}[]") != string::npos) {
                initmsg(eError) << "A condition containing an unevaluated "
                                << "formula {} or variable [] could not be interpreted." << eom;
                fAssertCondition = false;
            }
            else {
                if (aToken->GetValue<string>().empty()) {
                    fAssertCondition = false;
                }
                else {
                    fAssertCondition = aToken->GetValue<bool>();
                }
            }
            fAttributeState = eAttributeComplete;
            return;
        }
    }
    return;
}

void KPrintProcessor::ProcessToken(KEndAttributeToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementActive) {
        if (fAttributeState == eAttributeComplete) {
            fAttributeState = eAttributeInactive;
            return;
        }
    }
    return;
}

void KPrintProcessor::ProcessToken(KMidElementToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementActive) {
        fElementState = eElementComplete;
        return;
    }
    return;
}

void KPrintProcessor::ProcessToken(KElementDataToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementComplete) {
        initmsg(eError) << "got unknown element data <" << aToken->GetValue() << ">" << ret;
        initmsg(eError) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <"
                        << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
        return;
    }

    return;
}

void KPrintProcessor::ProcessToken(KEndElementToken* aToken)
{
    if (fElementState == eElementInactive) {
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementComplete) {
        if (!fCheckAssertCondition) {
            if (fName.empty()) {
                initmsg(fMessageType) << fValue;
            }
            else {
                initmsg(fMessageType) << "value of <" << fName << "> is <" << fValue << ">";
            }
        }
        else {
            if (fName.empty()) {
                initmsg(fMessageType) << "assertion is ";
            }
            else {
                initmsg(fMessageType) << "assertion of <" << fName << "> is ";
            }
            initmsg(fMessageType) << (fAssertCondition ? "true" : "false");

            if (!fAssertCondition) {
                fMessageType = eError;
            }
        }

        if (fMessageType == eError) {
            initmsg(fMessageType) << ret << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile()
                                  << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">"
                                  << eom;
        }
        else {
            initmsg(fMessageType) << eom;
        }

        fName.clear();
        fValue.clear();
        fAssertCondition = false;
        fCheckAssertCondition = false;
        fElementState = eElementInactive;
        fMessageType = eNormal;
        return;
    }

    return;
}

}  // namespace katrin
