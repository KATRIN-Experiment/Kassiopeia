#ifndef KMESSAGEBUILDER_H_
#define KMESSAGEBUILDER_H_

#include "KComplexElement.hh"
#include "KContainer.hh"
#include "KMessage.h"
#include "KTextFile.h"

namespace katrin
{

class KMessageData
{
  public:
    KMessageData();
    ~KMessageData();

  public:
    std::string fKey;
    KMessageFormat fFormat;
    KMessagePrecision fPrecision;
    bool fShowShutdownMessage;
    bool fShowParserContext;
    KMessageSeverity fTerminalVerbosity;
    KMessageSeverity fLogVerbosity;
};

typedef KComplexElement<KMessageData> KMessageDataBuilder;

template<> inline bool KMessageDataBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "key") {
        aContainer->CopyTo(fObject->fKey);
        return true;
    }
    if (aContainer->GetName() == "terminal") {
        if (aContainer->AsString() == "error") {
            fObject->fTerminalVerbosity = eError;
            return true;
        }
        if (aContainer->AsString() == "warning") {
            fObject->fTerminalVerbosity = eWarning;
            return true;
        }
        if (aContainer->AsString() == "normal") {
            fObject->fTerminalVerbosity = eNormal;
            return true;
        }
        if (aContainer->AsString() == "info") {
            fObject->fTerminalVerbosity = eInfo;
            return true;
        }
        if (aContainer->AsString() == "debug") {
            fObject->fTerminalVerbosity = eDebug;
            return true;
        }
        return false;
    }
    if (aContainer->GetName() == "log") {
        if (aContainer->AsString() == "error") {
            fObject->fLogVerbosity = eError;
            return true;
        }
        if (aContainer->AsString() == "warning") {
            fObject->fLogVerbosity = eWarning;
            return true;
        }
        if (aContainer->AsString() == "normal") {
            fObject->fLogVerbosity = eNormal;
            return true;
        }
        if (aContainer->AsString() == "info") {
            fObject->fLogVerbosity = eInfo;
            return true;
        }
        if (aContainer->AsString() == "debug") {
            fObject->fLogVerbosity = eDebug;
            return true;
        }
        return false;
    }
    if (aContainer->GetName() == "format") {
        if (aContainer->AsString() == "fixed") {
            fObject->fFormat = std::ios_base::fixed;
            return true;
        }
        if (aContainer->AsString() == "scientific") {
            fObject->fFormat = std::ios_base::scientific;
            return true;
        }
        return false;
    }
    if (aContainer->GetName() == "precision") {
        fObject->fPrecision = aContainer->AsReference<KMessagePrecision>();
        return true;
    }
    if (aContainer->GetName() == "shutdown_message") {
        fObject->fShowShutdownMessage = aContainer->AsReference<bool>();
        return true;
    }
    return false;
}


using KMessageTableBuilder = KComplexElement<KMessageTable>;

template<> inline bool KMessageTableBuilder::Begin()
{
    fObject = &KMessageTable::GetInstance();
    fObject->SetShowShutdownMessage();  // enabled by default
    return true;
}

template<> inline bool KMessageTableBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "terminal") {
        if (anAttribute->AsString() == "error") {
            fObject->SetTerminalVerbosity(eError);
            return true;
        }
        if (anAttribute->AsString() == "warning") {
            fObject->SetTerminalVerbosity(eWarning);
            return true;
        }
        if (anAttribute->AsString() == "normal") {
            fObject->SetTerminalVerbosity(eNormal);
            return true;
        }
        if (anAttribute->AsString() == "info") {
            fObject->SetTerminalVerbosity(eInfo);
            return true;
        }
        if (anAttribute->AsString() == "debug") {
            fObject->SetTerminalVerbosity(eDebug);
            return true;
        }
        return false;
    }
    if (anAttribute->GetName() == "log") {
        if (anAttribute->AsString() == "error") {
            fObject->SetLogVerbosity(eError);
            return true;
        }
        if (anAttribute->AsString() == "warning") {
            fObject->SetLogVerbosity(eWarning);
            return true;
        }
        if (anAttribute->AsString() == "normal") {
            fObject->SetLogVerbosity(eNormal);
            return true;
        }
        if (anAttribute->AsString() == "info") {
            fObject->SetLogVerbosity(eInfo);
            return true;
        }
        if (anAttribute->AsString() == "debug") {
            fObject->SetLogVerbosity(eDebug);
            return true;
        }
        return false;
    }
    if (anAttribute->GetName() == "format") {
        if (anAttribute->AsString() == "fixed") {
            fObject->SetFormat(std::ios_base::fixed);
            return true;
        }
        if (anAttribute->AsString() == "scientific") {
            fObject->SetFormat(std::ios_base::scientific);
            return true;
        }
        return false;
    }
    if (anAttribute->GetName() == "precision") {
        fObject->SetPrecision(anAttribute->AsReference<KMessagePrecision>());
        return true;
    }
    if (anAttribute->GetName() == "shutdown_message") {
        fObject->SetShowShutdownMessage(anAttribute->AsReference<bool>());
        return true;
    }
    if (anAttribute->GetName() == "parser_context") {
        fObject->SetShowParserContext(anAttribute->AsReference<bool>());
        return true;
    }
    return false;
}

template<> inline bool KMessageTableBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "file") {
        KTextFile* tFile = nullptr;
        anElement->ReleaseTo(tFile);

        tFile->SetDefaultPath(LOG_DEFAULT_DIR);
        tFile->SetDefaultBase("KasperLog.txt");
        tFile->Open(KFile::eWrite);
        if (tFile->IsOpen() == true) {
            fObject->SetLogStream(tFile->File());
        }
        else {
            KMessage tMessage("k_common", "KOMMON", "", "");
            tMessage(eWarning) << "could not open logfile" << eom;
        }
        return true;
    }
    if (anElement->GetName() == "message") {
        auto* tMessageData = anElement->AsPointer<KMessageData>();
        if (tMessageData->fKey == "all") {
            KMessageTable::GetInstance().SetTerminalVerbosity(tMessageData->fTerminalVerbosity);
            KMessageTable::GetInstance().SetLogVerbosity(tMessageData->fLogVerbosity);
            KMessageTable::GetInstance().SetFormat(tMessageData->fFormat);
            KMessageTable::GetInstance().SetPrecision(tMessageData->fPrecision);
            KMessageTable::GetInstance().SetShowShutdownMessage(tMessageData->fShowShutdownMessage);
            KMessageTable::GetInstance().SetShowParserContext(tMessageData->fShowParserContext);
            return true;
        }
        auto tMessage = KMessageTable::GetInstance().Get(tMessageData->fKey);
        if (tMessage) {
            tMessage->SetTerminalVerbosity(tMessageData->fTerminalVerbosity);
            tMessage->SetLogVerbosity(tMessageData->fLogVerbosity);
            tMessage->SetFormat(tMessageData->fFormat);
            tMessage->SetPrecision(tMessageData->fPrecision);
            tMessage->SetShowShutdownMessage(tMessageData->fShowShutdownMessage);
            tMessage->SetShowParserContext(tMessageData->fShowParserContext);
            return true;
        }
        else {
            KMessage tMessage("k_common", "KOMMON", "", "");
            tMessage(eWarning) << "no message registered with key <" << tMessageData->fKey << ">" << eom;
        }
        return true;
    }
    return false;
}

template<> inline bool KMessageTableBuilder::End()
{
    fObject = nullptr;
    return true;
}

}  // namespace katrin

#endif
