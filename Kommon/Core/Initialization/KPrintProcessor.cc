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
    fName(""),
    fValue("")
{}

KPrintProcessor::~KPrintProcessor() {}

void KPrintProcessor::ProcessToken(KBeginElementToken* aToken)
{
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
        if (aToken->GetValue() == "value") {
            fAttributeState = eActiveValue;
            return;
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
        if (fName.empty()) {
            initmsg(fMessageType) << fValue;
        }
        else {
            initmsg(fMessageType) << "value of <" << fName << "> is <" << fValue << ">";
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
        fElementState = eElementInactive;
        fMessageType = eNormal;
        return;
    }

    return;
}

}  // namespace katrin
