#include "KSerializationProcessor.hh"

#include "KInitializationMessage.hh"

#include <fstream>
#include <typeinfo>

using namespace std;

inline string addIndent(int n)
{
    string out;
    for (int i = 0; i < n; ++i)
        out += "  ";
    return out;
}

namespace katrin
{

KSerializationProcessor::KSerializationProcessor() :
    fIndentLevel(0),
    fXmlConfig(""),
    fYamlConfig(""),
    fJsonConfig(""),
    fOutputFilename(""),
    fElementName(""),
    fIsChildElement(false),
    fAttributeCount(0),
    fElementState(eElementInactive),
    fAttributeState(eAttributeInactive)
{}
KSerializationProcessor::~KSerializationProcessor() = default;

void KSerializationProcessor::ProcessToken(KBeginParsingToken* aToken)
{
    fXmlConfig += "<!-- got a begin parsing token -->\n";

    initmsg_debug("<!-- got a begin parsing token -->" << eom);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KBeginFileToken* aToken)
{
    fXmlConfig += "<!-- <file path=\"";
    fXmlConfig += aToken->GetValue();
    fXmlConfig += "\"> -->\n";

    fYamlConfig += "# file path: ";
    fYamlConfig += aToken->GetValue();
    fYamlConfig += "\n";

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

        fXmlConfig += addIndent(fIndentLevel);
        fXmlConfig += "<";
        fXmlConfig += aToken->GetValue();

        if (!fJsonConfig.empty() && fJsonConfig.back() == '}')
            fJsonConfig += ",\n";

        if (aToken->GetValue() != fElementName || fIsChildElement) {
            fYamlConfig += addIndent(fIndentLevel);
            fYamlConfig += "- ";
            fYamlConfig += aToken->GetValue();
            fYamlConfig += ":\n";

            fJsonConfig += addIndent(fIndentLevel);
            fJsonConfig += "{ \"";
            fJsonConfig += aToken->GetValue();
            fJsonConfig += "\": [\n";
        }

        fElementName = aToken->GetValue();
        fAttributeCount = 0;
        fIndentLevel++;

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

    fXmlConfig += " ";
    fXmlConfig += aToken->GetValue();
    fXmlConfig += "=\"";

    if (fAttributeName == aToken->GetValue()) {
        fYamlConfig += addIndent(fIndentLevel-1);
        fYamlConfig += "- ";
    }
    else {
        fYamlConfig += addIndent(fIndentLevel);
    }

    if (fAttributeCount == 0) {
        fYamlConfig += "- ";

        fJsonConfig += addIndent(fIndentLevel);
        fJsonConfig += "{\n";

        fIndentLevel++;
    }
    fYamlConfig += "_";
    fYamlConfig += aToken->GetValue();
    fYamlConfig += ": ";

    if (fAttributeCount > 0)
        fJsonConfig += ",\n";
    fJsonConfig += addIndent(fIndentLevel);
    fJsonConfig += "\"_";
    fJsonConfig += aToken->GetValue();
    fJsonConfig += "\": \"";

    fAttributeName = aToken->GetValue();
    fAttributeCount++;

    initmsg_debug(" " << aToken->GetValue() << "=\"");
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KAttributeDataToken* aToken)
{
    if (fElementState == eElementInactive) {
        fXmlConfig += aToken->GetValue();

        fYamlConfig += aToken->GetValue();

        fJsonConfig += aToken->GetValue();

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
        fXmlConfig += "\"";

        fYamlConfig += "\n";

        fJsonConfig += "\"";

        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eActiveFileDefine) {
        if (fAttributeState == eAttributeComplete) {
            fAttributeState = eAttributeInactive;
            return;
        }
    }

    if (fAttributeState == eAttributeComplete)
        fAttributeState = eAttributeInactive;

    initmsg_debug("\"");
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    return;
}
void KSerializationProcessor::ProcessToken(KMidElementToken* aToken)
{
    if (fElementState == eElementInactive) {
        fXmlConfig += ">\n";

        if (fAttributeCount > 0) {
            fJsonConfig += "\n";
            fJsonConfig += addIndent(fIndentLevel-1);
            fJsonConfig += "}";

            fIndentLevel--;
        }

        fAttributeName = "";
        fIsChildElement = true;

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
    fIndentLevel--;

    if (fElementState == eElementInactive) {
        fXmlConfig += addIndent(fIndentLevel);
        fXmlConfig += "</";
        fXmlConfig += aToken->GetValue();
        fXmlConfig += ">\n";

        fJsonConfig += "\n";
        fJsonConfig += addIndent(fIndentLevel);
        fJsonConfig += "]}";

        fElementName = "";
        fIsChildElement = false;

        KProcessor::ProcessToken(aToken);
        return;
    }

    if (fElementState == eElementComplete)
        fElementState = eElementInactive;

    initmsg_debug("    </" << aToken->GetValue() << ">" << ret << ret);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    return;
}
void KSerializationProcessor::ProcessToken(KEndFileToken* aToken)
{
    fXmlConfig += "<!-- </file> -->\n";

    initmsg_debug("<!-- </file> -->" << ret);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KEndParsingToken* aToken)
{
    fXmlConfig += "<!-- got an end parsing token -->\n";

    initmsg_debug("<!-- got an end parsing token -->" << eom);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KCommentToken* aToken)
{
    fXmlConfig += "\n<!--";
    fXmlConfig += aToken->GetValue();
    fXmlConfig += "-->\n";

    // FIXME: yaml does not support multi-line comments

    initmsg_debug("<!--" << aToken->GetValue() << "-->" << ret);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}
void KSerializationProcessor::ProcessToken(KErrorToken* aToken)
{
    fXmlConfig += "<!-- got an error token <";
    fXmlConfig += aToken->GetValue();
    fXmlConfig += ">-->\n";

    fYamlConfig += "# error: ";
    fYamlConfig += aToken->GetValue();
    fYamlConfig += "\n";

    initmsg_debug("<!-- got an error token <" << aToken->GetValue() << "> -->" << eom);
    initmsg_debug("at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
    KProcessor::ProcessToken(aToken);
    return;
}

}  // namespace katrin
