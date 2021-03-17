#include "KSerializationProcessor.hh"

#include "KInitializationMessage.hh"

#include <fstream>
#include <typeinfo>

using namespace std;

namespace katrin
{

KSerializationProcessor::KSerializationProcessor() :
    completeconfig(""),
    fOutputFilename(""),
    fElementState(eElementInactive),
    fAttributeState(eAttributeInactive)
{}
KSerializationProcessor::~KSerializationProcessor() = default;

void KSerializationProcessor::ProcessToken(KBeginParsingToken* aToken)
{
    completeconfig += "<!-- got a begin parsing token -->\n";

    initmsg_debug("<!-- got a begin parsing token -->" << eom);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KBeginFileToken* aToken)
{
    completeconfig += "<!-- <file path=\"";
    completeconfig += aToken->GetValue();
    completeconfig += "\"> -->\n";

    initmsg_debug("<!-- <file path=\"" << aToken->GetValue() << "\"> -->" << ret);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KBeginElementToken* aToken)
{
    if (fElementState == eElementInactive) {
        if (aToken->GetValue() == string("serialization")) {
            fElementState = eActiveFileDefine;
            return;
        }

        completeconfig += "    <";
        completeconfig += aToken->GetValue();
        KProcessor::ProcessToken(aToken);
        return;
    }

    initmsg_debug("    <" << aToken->GetValue());
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    initmsg(eError) << "KSerializationProcessor: got unknown element <" << aToken->GetValue() << ">" << ret;
    initmsg(eError) << "KSerializationProcessor: in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile()
                    << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
    return;
}
void KSerializationProcessor::ProcessToken(KBeginAttributeToken* aToken)
{
    if (fElementState == eActiveFileDefine) {
        if (aToken->GetValue() == "file") {
            if (fOutputFilename.size() == 0) {
                fAttributeState = eActiveFileName;
                return;
            }
            else {
                initmsg << "KSerializationProcessor: file attribute must appear only once in definition" << ret;
                initmsg(eError) << "KSerializationProcessor: in path <" << aToken->GetPath() << "> in file <"
                                << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <"
                                << aToken->GetColumn() << ">" << eom;
                return;
            }
        }

        initmsg(eError) << "KSerializationProcessor: got unknown attribute <" << aToken->GetValue() << ">" << ret;
        initmsg(eError) << "KSerializationProcessor: in path <" << aToken->GetPath() << "> in file <"
                        << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <"
                        << aToken->GetColumn() << ">" << eom;

        return;
    }

    completeconfig += " ";
    completeconfig += aToken->GetValue();
    completeconfig += "=\"";
    initmsg_debug(" " << aToken->GetValue() << "=\"");
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KAttributeDataToken* aToken)
{

    if (fElementState == eElementInactive) {
        completeconfig += aToken->GetValue();
        KProcessor::ProcessToken(aToken);
        return;
    }
    if (fElementState == eActiveFileDefine) {
        if (fAttributeState == eActiveFileName) {
            fOutputFilename = aToken->GetValue();
            fAttributeState = eAttributeComplete;
            return;
        }
    }

    initmsg_debug(aToken->GetValue());
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    return;
}
void KSerializationProcessor::ProcessToken(KEndAttributeToken* aToken)
{

    if (fElementState == eElementInactive) {
        completeconfig += "\"";
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActiveFileDefine) {
        if (fAttributeState == eAttributeComplete) {
            fAttributeState = eAttributeInactive;
            return;
        }
    }

    initmsg_debug("\"");
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    return;
}
void KSerializationProcessor::ProcessToken(KMidElementToken* aToken)
{

    if (fElementState == eElementInactive) {
        completeconfig += ">\n";
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActiveFileDefine) {
        fElementState = eElementComplete;
    }

    initmsg_debug(">" << ret);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KElementDataToken* aToken)
{
    initmsg_debug("got an element data token <" << aToken->GetValue() << ">" << eom);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);

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
void KSerializationProcessor::ProcessToken(KEndElementToken* aToken)
{

    if (fElementState == eElementInactive) {
        completeconfig += "    </";
        completeconfig += aToken->GetValue();
        completeconfig += ">\n\n";
        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementComplete) {
        fElementState = eElementInactive;
        return;
    }

    initmsg_debug("    </" << aToken->GetValue() << ">" << ret << ret);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    return;
}
void KSerializationProcessor::ProcessToken(KEndFileToken* aToken)
{
    completeconfig += "<!-- </file> -->\n";

    initmsg_debug("<!-- </file> -->" << ret);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KEndParsingToken* aToken)
{
    completeconfig += "<!-- got an end parsing token -->\n";

    initmsg_debug("<!-- got an end parsing token -->" << eom);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KCommentToken* aToken)
{
    completeconfig += "\n<!--";
    completeconfig += aToken->GetValue();
    completeconfig += "-->\n";

    initmsg_debug("<!--" << aToken->GetValue() << "-->" << ret);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KErrorToken* aToken)
{
    completeconfig += "<!-- got an error token <";
    completeconfig += aToken->GetValue();
    completeconfig += ">-->\n";

    initmsg_debug("<!-- got an error token <" << aToken->GetValue() << "> -->" << eom);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}

}  // namespace katrin
