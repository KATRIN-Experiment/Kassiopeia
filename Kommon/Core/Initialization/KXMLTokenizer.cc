#include "KXMLTokenizer.hh"

#include "KInitializationMessage.hh"
#include "KLogger.h"
KLOGGER("kommon.init");

#include <chrono>
#include <sstream>
#include <cmath>

using namespace std;

namespace katrin
{

KXMLTokenizer::KXMLTokenizer() :
    KProcessor(),
    fFile(nullptr),
    fPath(""),
    fName(""),
    fLine(0),
    fColumn(0),
    fChar('\0'),

    fState(),
    fInitialState(&KXMLTokenizer::ParseBegin),
    fFinalState(&KXMLTokenizer::ParseEnd),

    fNestedCommentCounter(0),

    fBuffer(""),
    fNames(),
    fBeginParsing(new KBeginParsingToken()),
    fBeginFile(new KBeginFileToken()),
    fBeginElement(new KBeginElementToken()),
    fBeginAttribute(new KBeginAttributeToken()),
    fAttributeData(new KAttributeDataToken()),
    fEndAttribute(new KEndAttributeToken()),
    fMidElement(new KMidElementToken()),
    fElementData(new KElementDataToken()),
    fEndElement(new KEndElementToken()),
    fEndFile(new KEndFileToken()),
    fEndParsing(new KEndParsingToken()),
    fComment(new KCommentToken()),
    fError(new KErrorToken())
{}

KXMLTokenizer::~KXMLTokenizer()
{
    delete fBeginParsing;
    delete fBeginFile;
    delete fBeginElement;
    delete fBeginAttribute;
    delete fAttributeData;
    delete fEndAttribute;
    delete fMidElement;
    delete fElementData;
    delete fEndElement;
    delete fEndFile;
    delete fEndParsing;
    delete fComment;
    delete fError;
}

void KXMLTokenizer::ProcessFile(KTextFile* aFile)
{
    // keep track of time spent on parsing files
    std::chrono::steady_clock::time_point tClockStart, tClockEnd;
    std::chrono::milliseconds tElapsedTime;

    fFile = aFile;
    fState = fInitialState;
    while (true) {
        if (fState == &KXMLTokenizer::ParseBeginFile) {
            KDEBUG("Parsing: " << fFile->GetName());
            tClockStart = std::chrono::steady_clock::now();
        }

        ((this)->*(fState))();

        if (fState == &KXMLTokenizer::ParseEndFile) {
            tClockEnd = std::chrono::steady_clock::now();
            tElapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(tClockEnd - tClockStart);
            tClockStart = tClockEnd;

            auto tMilliSeconds = tElapsedTime.count();
            KDEBUG("Finished: " << fFile->GetName() << " (took " << tMilliSeconds << " ms)");

            if (tMilliSeconds > 10000)
                initmsg(eWarning) << "It took " << ceil(tMilliSeconds / 100.) / 10. << " s to process the file <"
                                  << fFile->GetName() << ">" << eom;
        }

        if (fState == fFinalState) {
            ((this)->*(fState))();
            break;
        }
    }
    return;
}

void KXMLTokenizer::ParseBegin()
{
    //initmsg_debug( "in begin" << eom );

    ProcessToken(fBeginParsing);
    fState = &KXMLTokenizer::ParseBeginFile;

    return;
}
void KXMLTokenizer::ParseBeginFile()
{
    //initmsg_debug( "in beginfile" << eom );

    //if file does not open then bail, otherwise go on
    if (fFile->Open(KFile::eRead) == false) {
        fBuffer = string("unable to open file with name <") + fFile->GetName() + string(">");
        fError->SetValue(fBuffer);
        ProcessToken(fError);
        fState = fFinalState;
        return;
    }
    else {
        fPath = fFile->GetPath();
        fName = fFile->GetName();
        fLine = 1;
        fColumn = 1;
        fChar = fFile->File()->get();

        fBeginFile->SetPath(fPath);
        fBeginElement->SetPath(fPath);
        fBeginAttribute->SetPath(fPath);
        fAttributeData->SetPath(fPath);
        fEndAttribute->SetPath(fPath);
        fMidElement->SetPath(fPath);
        fElementData->SetPath(fPath);
        fEndElement->SetPath(fPath);
        fEndFile->SetPath(fPath);
        fComment->SetPath(fPath);
        fError->SetPath(fPath);

        fBeginFile->SetFile(fName);
        fBeginElement->SetFile(fName);
        fBeginAttribute->SetFile(fName);
        fAttributeData->SetFile(fName);
        fEndAttribute->SetFile(fName);
        fMidElement->SetFile(fName);
        fElementData->SetFile(fName);
        fEndElement->SetFile(fName);
        fEndFile->SetFile(fName);
        fComment->SetFile(fName);
        fError->SetFile(fName);

        fBeginFile->SetValue(fFile->GetName());
        ProcessToken(fBeginFile);

        fBuffer.clear();

        fState = &KXMLTokenizer::ParseElementData;
        return;
    }
}
void KXMLTokenizer::ParseElementBeginName()
{
    //initmsg_debug( "in elementbeginname: current character is <" << fChar << ">" << eom )

    //if at name, append char to name, then recurse
    if (AtOneOf(fNameChars)) {
        fBuffer.append(1, fChar);

        Increment();

        fState = &KXMLTokenizer::ParseElementBeginName;
        return;
    }

    //if at "/>", then send start element, then send mid and end element, then ParseElementData
    if (AtExactly(fRightSlashAngle)) {
        fBeginElement->SetValue(fBuffer);
        ProcessToken(fBeginElement);

        fMidElement->SetLine(fLine);
        fMidElement->SetColumn(fColumn);
        fMidElement->SetValue(fBuffer);
        ProcessToken(fMidElement);

        fEndElement->SetLine(fLine);
        fEndElement->SetColumn(fColumn);
        fEndElement->SetValue(fBuffer);
        ProcessToken(fEndElement);

        fBuffer.clear();
        Increment();
        Increment();

        fState = &KXMLTokenizer::ParseElementData;
        return;
    }

    //if at ">", the put name on stack, then send start element, then send mid element, then ParseElementData
    if (AtExactly(fRightAngle)) {
        fBeginElement->SetValue(fBuffer);
        ProcessToken(fBeginElement);

        fMidElement->SetLine(fLine);
        fMidElement->SetColumn(fColumn);
        fMidElement->SetValue(fBuffer);
        ProcessToken(fMidElement);

        fNames.push(fBuffer);
        fBuffer.clear();
        Increment();

        fState = &KXMLTokenizer::ParseElementData;
        return;
    }

    //if at whitespace, then put name on stack, then send start element, then ParseElementHeader
    if (AtOneOf(fWhiteSpaceChars)) {
        fBeginElement->SetValue(fBuffer);
        ProcessToken(fBeginElement);

        fNames.push(fBuffer);
        fBuffer.clear();
        Increment();

        fState = &KXMLTokenizer::ParseElementHeader;
        return;
    }

    fBuffer = string("got unknown character <") + fChar + string(">");
    fError->SetValue(fBuffer);
    fError->SetLine(fLine);
    fError->SetColumn(fColumn);
    ProcessToken(fError);
    fState = fFinalState;

    return;
}
void KXMLTokenizer::ParseElementHeader()
{
    //initmsg_debug( "in elementheader: current character is <" << fChar << ">" << eom )

    //if at whitespace, then recurse
    if (AtOneOf(fWhiteSpaceChars)) {
        Increment();

        fState = &KXMLTokenizer::ParseElementHeader;
        return;
    }

    //if at "/>", then send mid element, then send end element, then pop stack and clear buffer, then ParseElementData
    if (AtExactly(fRightSlashAngle)) {
        fMidElement->SetLine(fLine);
        fMidElement->SetColumn(fColumn);
        fMidElement->SetValue(fNames.top());
        ProcessToken(fMidElement);

        fEndElement->SetLine(fLine);
        fEndElement->SetColumn(fColumn);
        fEndElement->SetValue(fNames.top());
        ProcessToken(fEndElement);

        fNames.pop();
        fBuffer.clear();
        Increment();
        Increment();

        fState = &KXMLTokenizer::ParseElementData;
        return;
    }

    //if at ">", then send mid element, then clear buffer, then ParseElementData
    if (AtOneOf(fRightAngle)) {
        fMidElement->SetLine(fLine);
        fMidElement->SetColumn(fColumn);
        fMidElement->SetValue(fNames.top());
        ProcessToken(fMidElement);

        fBuffer.clear();
        Increment();

        fState = &KXMLTokenizer::ParseElementData;
        return;
    }

    //if at name start, then ParseAttributeName
    if (AtOneOf(fNameStartChars)) {
        fBeginAttribute->SetLine(fLine);
        fBeginAttribute->SetColumn(fColumn);

        fBuffer.clear();
        //do NOT increment!

        fState = &KXMLTokenizer::ParseAttributeName;
        return;
    }

    fBuffer = string("got unknown character <") + fChar + string(">");
    fError->SetValue(fBuffer);
    fError->SetLine(fLine);
    fError->SetColumn(fColumn);
    ProcessToken(fError);
    fState = fFinalState;
    return;
}
void KXMLTokenizer::ParseElementData()
{
    //initmsg_debug( "in elementvalue: current character is <" << fChar << ">" << eom )

    //if at whitespace or value, then append character to value, recurse
    if (AtOneOf(fWhiteSpaceChars) || AtOneOf(fValueChars)) {
        fBuffer.append(1, fChar);

        Increment();
        fState = &KXMLTokenizer::ParseElementData;
        return;
    }

    //if at "<!-- then prepare comment, then ParseComment
    if (AtExactly(fCommentStart)) {
        fComment->SetLine(fLine);
        fComment->SetColumn(fColumn);
        fNestedCommentCounter = 1;

        fCommentBuffer.clear();
        Increment();
        Increment();
        Increment();
        Increment();

        fState = &KXMLTokenizer::ParseComment;
        return;
    }

    //if at "</", then send data, then ParseEndElementName
    if (AtExactly(fLeftAngleSlash)) {
        fBuffer = Trim(fBuffer);
        fElementData->SetValue(fBuffer);
        if (fBuffer.size() != 0) {
            ProcessToken(fElementData);
        }
        fBuffer.clear();

        fEndElement->SetLine(fLine);
        fEndElement->SetColumn(fColumn);

        Increment();
        Increment();

        fState = &KXMLTokenizer::ParseElementEndName;
        return;
    }

    //if at "<", then send data, then ParseStartElementName
    if (AtExactly(fLeftAngle)) {
        fBuffer = Trim(fBuffer);
        fElementData->SetValue(fBuffer);
        if (fBuffer.size() != 0) {
            ProcessToken(fElementData);
        }
        fBuffer.clear();

        fBeginElement->SetLine(fLine);
        fBeginElement->SetColumn(fColumn);

        Increment();

        fState = &KXMLTokenizer::ParseElementBeginName;
        return;
    }

    //if at end of file, then ParseEndFile
    if (AtEnd()) {
        fBuffer = Trim(fBuffer);
        fElementData->SetValue(fBuffer);
        if (fBuffer.size() != 0) {
            ProcessToken(fElementData);
        }
        fBuffer.clear();

        fState = &KXMLTokenizer::ParseEndFile;
        return;
    }

    fBuffer = string("got unknown character <") + fChar + string(">");
    fError->SetValue(fBuffer);
    fError->SetLine(fLine);
    fError->SetColumn(fColumn);
    ProcessToken(fError);
    fState = fFinalState;
    return;
}
void KXMLTokenizer::ParseElementEndName()
{
    //initmsg_debug( "in endelementname: current character is <" << fChar << ">" << eom )

    //if at name, then append char to name, then recurse
    if (AtOneOf(fNameChars)) {
        fBuffer.append(1, fChar);
        Increment();

        fState = &KXMLTokenizer::ParseElementEndName;
        return;
    }

    //if at ">", then check and send end element, then prepare value, then ParseBody
    if (AtOneOf(fRightAngle)) {
        if (fNames.top() != fBuffer) {
            fBuffer = string("expected closing element name <") + fNames.top() + string(">, but got <") + fBuffer +
                      string("> instead");
            fError->SetValue(fBuffer);
            fError->SetLine(fLine);
            fError->SetColumn(fColumn);
            ProcessToken(fError);
            fState = fFinalState;
            return;
        }
        fNames.pop();

        fEndElement->SetValue(fBuffer);
        ProcessToken(fEndElement);

        fBuffer.clear();
        Increment();

        fState = &KXMLTokenizer::ParseElementData;
        return;
    }

    fBuffer = string("got unknown character <") + fChar + string(">");
    fError->SetValue(fBuffer);
    fError->SetLine(fLine);
    fError->SetColumn(fColumn);
    ProcessToken(fError);
    fState = fFinalState;
    return;
}
void KXMLTokenizer::ParseAttributeName()
{
    //initmsg_debug( "in attributename: current character is <" << fChar << ">" << eom )

    //if at name, then append char to name buffer, then recurse
    if (AtOneOf(fNameChars)) {
        fBuffer.append(1, fChar);
        Increment();

        fState = &KXMLTokenizer::ParseAttributeName;
        return;
    }

    //if at '="', then ParseAttributeAssignementPost
    if (AtExactly(fEqual + fQuote)) {
        fNames.push(fBuffer);
        fBeginAttribute->SetValue(fBuffer);
        ProcessToken(fBeginAttribute);

        fAttributeData->SetLine(fLine);
        fAttributeData->SetColumn(fColumn);

        fBuffer.clear();
        Increment();
        Increment();

        fState = &KXMLTokenizer::ParseAttributeValue;
        return;
    }

    fBuffer = string("got unknown character <") + fChar + string(">");
    fError->SetValue(fBuffer);
    fError->SetLine(fLine);
    fError->SetColumn(fColumn);
    ProcessToken(fError);
    fState = fFinalState;
    return;
}
void KXMLTokenizer::ParseAttributeValue()
{
    //initmsg_debug( "in attributevalue; current character is <" << fChar << ">" << eom )

    //if at whitespace or value, then append char to value, then recurse
    if (AtOneOf(fWhiteSpaceChars) || AtOneOf(fValueChars)) {
        fBuffer.append(1, fChar);
        Increment();

        fState = &KXMLTokenizer::ParseAttributeValue;
        return;
    }

    //if at """, then send attribute value token, then send attribute end token, then ParseElementPrefix
    if (AtOneOf(fQuote)) {
        fBuffer = Trim(fBuffer);
        fAttributeData->SetValue(fBuffer);
        if (fBuffer.size() != 0) {
            ProcessToken(fAttributeData);
        }

        fEndAttribute->SetLine(fBeginAttribute->GetLine());
        fEndAttribute->SetColumn(fBeginAttribute->GetColumn());
        fEndAttribute->SetValue(fNames.top());
        ProcessToken(fEndAttribute);

        fBuffer.clear();
        fNames.pop();

        Increment();

        fState = &KXMLTokenizer::ParseElementHeader;
        return;
    }

    fBuffer = string("got unknown character <") + fChar + string(">");
    fError->SetValue(fBuffer);
    fError->SetLine(fLine);
    fError->SetColumn(fColumn);
    ProcessToken(fError);
    fState = fFinalState;
    return;
}
void KXMLTokenizer::ParseEndFile()
{
    //initmsg_debug( "in endfile" << eom );

    //if at whitespace, then append char to value, then recurse
    if (fFile->Close() == false) {
        fBuffer = string("unable to close file with key <") + fFile->GetName() + string(">");
        fError->SetValue(fBuffer);
        fError->SetLine(fLine);
        fError->SetColumn(fColumn);
        ProcessToken(fError);
        fState = fFinalState;
        return;
    }

    fEndFile->SetValue(fName);
    ProcessToken(fEndFile);

    fPath = string("");
    fName = string("");
    fLine = 0;
    fColumn = 0;
    fChar = '\0';

    fState = &KXMLTokenizer::ParseEnd;
    return;
}
void KXMLTokenizer::ParseEnd()
{
    //initmsg_debug( "in end: current file name is <" << fFile->GetName() << ">" << eom )

    ProcessToken(fEndParsing);
    fState = &KXMLTokenizer::ParseEnd;
    return;
}
void KXMLTokenizer::ParseComment()
{
    //initmsg_debug( "in comment: current character is <" << fChar << ">" << eom )

    //if at "-->" then prepare comment, then revert to old state
    if (AtExactly(fCommentEnd)) {
        Increment();
        Increment();
        Increment();

        --fNestedCommentCounter;
        //if comment counter is zero -> end comment
        if (fNestedCommentCounter == 0) {
            fComment->SetValue(fCommentBuffer);
            ProcessToken(fComment);
            fState = &KXMLTokenizer::ParseElementData;
        }
        else {
            fCommentBuffer.append(fCommentEnd);
        }
        return;
    }

    //if at "<!--" then increment comment counter
    if (AtExactly(fCommentStart)) {
        ++fNestedCommentCounter;
        fCommentBuffer.append(fCommentStart);
        Increment();
        Increment();
        Increment();
        Increment();
    }

    //if at end of file, then ParseEndFile
    if (AtEnd()) {
        fComment->SetValue(fCommentBuffer);
        ProcessToken(fComment);

        initmsg(eWarning) << "found <" << fNestedCommentCounter << "> unclosed comment(s) at end of file <"
                          << fFile->GetName() << ">" << eom;
        fNestedCommentCounter = 0;

        fState = &KXMLTokenizer::ParseEndFile;
        return;
    }

    //if at anything else, then append character to value, recurse

    fCommentBuffer.append(1, fChar);

    Increment();

    fState = &KXMLTokenizer::ParseComment;
    return;
}

void KXMLTokenizer::Increment()
{
    //if iterator is already the end, bail out
    if (AtEnd()) {
        return;
    }

    //calculate adjustments to the line and column numbers
    int ColumnChange;
    int LineChange;
    if (AtExactly(fNewLine)) {
        //if the current character is a newline, a successful increment will make the column number 1 and the line number jump by one.
        ColumnChange = 1 - fColumn;
        LineChange = 1;
    }
    else {
        //if the current character is not a newline, a successful increment will make the column number jump by one and the line number will stay the same.
        ColumnChange = 1;
        LineChange = 0;
    }

    //increment the iterator
    //initmsg_debug( "popping the iterator" << eom )
    fChar = fFile->File()->get();

    //make sure that incrementing didn't put the iterator at the end
    if (AtEnd()) {
        return;
    }

    //apply the calculated column and line adjustments
    fColumn = fColumn + ColumnChange;
    fLine = fLine + LineChange;

    //GET OUT
    return;
}
bool KXMLTokenizer::AtEnd()
{
    //if iterator is at EOF, return true
    if (fFile->File()->good() == false) {
        return true;
    }
    return false;
}
bool KXMLTokenizer::AtOneOf(const string& chars)
{
    //if iterator is at EOF, return false
    if (AtEnd()) {
        return false;
    }

    //if iterator is at a character contained in chars, return true
    if (chars.find(fChar) != string::npos) {
        return true;
    }

    //otherwise return false
    return false;
}
bool KXMLTokenizer::AtExactly(const string& aString)
{
    //if iterator is already at EOF, return false
    if (AtEnd()) {
        return false;
    }

    //match the string
    bool tMatched = true;
    string::const_iterator tIter = aString.begin();
    while (tIter != aString.end()) {
        //grab a character
        //initmsg_debug( "file character is <" << fChar << ">, comparison character is <" << *tIter << ">" << eom )

        //if that puts us at the end, try to go back and return false
        if (AtEnd()) {
            //initmsg_debug( "hit the end" << eom )
            tMatched = false;
            break;
        }
        //if a mismatch is found, try to go back and return false
        if (fChar != *tIter) {
            //initmsg_debug( "<" << fChar << "> seriously does not match <" << *tIter << ">" << eom )
            tMatched = false;
            break;
        }

        //initmsg_debug( "popping the iterator" << eom )

        fChar = fFile->File()->get();
        tIter++;
    }

    //go back
    while (tIter != aString.begin()) {
        //initmsg_debug( "unpopping the iterator" << eom )

        fFile->File()->unget();
        tIter--;
    }

    //reset to character that we were originally on
    fFile->File()->unget();
    fChar = fFile->File()->get();

    return tMatched;
}

string KXMLTokenizer::Trim(const string& aBuffer)
{
    //        const size_t tBufferSize = aBuffer.size();
    //        const char* tBufferArray = aBuffer.c_str();
    //
    //        if( tBufferSize == 0 )
    //        {
    //            return string();
    //        }
    //
    //        const char* const tFirst = tBufferArray[0];
    //        const char* const tLast = tBufferArray[tBufferSize - 1];
    //
    //        const char* tProcessedFirst = tFirst;
    //        const char* tProcessedLast = tLast;
    //
    //        while( tProcessedFirst != tLast )
    //        {
    //            if( fWhiteSpaceChars.find( *tProcessedFirst ) == string::npos )
    //            {
    //                break;
    //            }
    //            tProcessedFirst++;
    //        }
    //
    //        while( tProcessedLast != tFirst )
    //        {
    //            if( fWhiteSpaceChars.find( *tProcessedLast ) == string::npos )
    //            {
    //                break;
    //            }
    //            tProcessedLast--;
    //        }
    //
    //        if( tProcessedFirst > tProcessedLast )
    //        {
    //            return string();
    //        }
    //
    //        return string( tProcessedFirst, tProcessedLast - tProcessedFirst + 1 );

    string tReturn("");

    if (aBuffer.size() == 0) {
        return tReturn;
    }

    string::const_iterator tBegin = aBuffer.begin();
    string::const_iterator tEnd = aBuffer.begin() + aBuffer.size() - 1;

    string::const_iterator tTrimmedBegin = tBegin;
    while (tTrimmedBegin != tEnd) {
        if (fWhiteSpaceChars.find(*tTrimmedBegin) == string::npos) {
            break;
        }
        tTrimmedBegin++;
    }

    string::const_iterator tTrimmedEnd = tEnd;
    while (tTrimmedEnd != tBegin) {
        if (fWhiteSpaceChars.find(*tTrimmedEnd) == string::npos) {
            break;
        }
        tTrimmedEnd--;
    }

    if (tTrimmedEnd < tTrimmedBegin) {
        return tReturn;
    }

    string::const_iterator tIter;
    bool tFlag = false;
    for (tIter = tTrimmedBegin; tIter <= tTrimmedEnd; tIter++) {
        if (fWhiteSpaceChars.find(*tIter) != string::npos) {
            if (tFlag == false) {
                tFlag = true;
                tReturn.append(1, ' ');
            }
            continue;
        }
        tFlag = false;
        tReturn.append(1, *tIter);
    }

    return tReturn;
}

const string KXMLTokenizer::fSpace = string(" ");
const string KXMLTokenizer::fTab = string("\t");
const string KXMLTokenizer::fNewLine = string("\n");
const string KXMLTokenizer::fCarriageReturn = string("\r");

const string KXMLTokenizer::fLowerCase = string("abcdefghijklmnopqrstuvwxyz");
const string KXMLTokenizer::fUpperCase = string("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
const string KXMLTokenizer::fNumerals = string("0123456789");

const string KXMLTokenizer::fLeftAngle = string("<");
const string KXMLTokenizer::fRightAngle = string(">");
const string KXMLTokenizer::fLeftAngleSlash = string("</");
const string KXMLTokenizer::fRightSlashAngle = string("/>");
const string KXMLTokenizer::fEqual = string("=");
const string KXMLTokenizer::fQuote = string("\"");
const string KXMLTokenizer::fCommentStart = string("<!--");
const string KXMLTokenizer::fCommentEnd = string("-->");

const string KXMLTokenizer::fLeftBraces = string("([");
const string KXMLTokenizer::fOperators = string("+-*/&|!");
const string KXMLTokenizer::fRightBraces = string(")]");

const string KXMLTokenizer::fParameterLeftBrace = string("{");
const string KXMLTokenizer::fParameterRightBrace = string("}");

const string KXMLTokenizer::fWhiteSpaceChars =
    KXMLTokenizer::fSpace + KXMLTokenizer::fTab + KXMLTokenizer::fNewLine + KXMLTokenizer::fCarriageReturn;
const string KXMLTokenizer::fNameStartChars = KXMLTokenizer::fLowerCase + KXMLTokenizer::fUpperCase + string(":_");
const string KXMLTokenizer::fNameChars =
    KXMLTokenizer::fLowerCase + KXMLTokenizer::fUpperCase + KXMLTokenizer::fNumerals + string(":_-.");
const string KXMLTokenizer::fValueChars =
    KXMLTokenizer::fLowerCase + KXMLTokenizer::fUpperCase + KXMLTokenizer::fNumerals + KXMLTokenizer::fLeftBraces +
    KXMLTokenizer::fOperators + KXMLTokenizer::fRightBraces + KXMLTokenizer::fParameterLeftBrace +
    KXMLTokenizer::fParameterRightBrace + string("@:_.,#;^");
}  // namespace katrin
