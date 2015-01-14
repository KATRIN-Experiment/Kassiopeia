#ifndef Kommon_KXMLTokenizer_hh_
#define Kommon_KXMLTokenizer_hh_

#include "KProcessor.hh"

#include "KFile.h"
using katrin::KFile;

#include "KTextFile.h"
using katrin::KTextFile;

#include <stack>
using std::stack;

#include <vector>
using std::vector;

#include <cstdlib>

namespace katrin
{

    //this is a shitty xml lexer.
    class KXMLTokenizer :
        public KProcessor
    {
        public:
            KXMLTokenizer();
            virtual ~KXMLTokenizer();

            //**********
            //processing
            //**********

        public:
            void ProcessFile( KTextFile* aFile );

            //*******
            //queuing
            //*******

        private:
            void Increment();
            bool AtEnd();
            bool AtOneOf( const string& aSet );
            bool AtExactly( const string& aString );

            KTextFile* fFile;
            string fPath;
            string fName;
            int fLine;
            int fColumn;
            char fChar;

            //*******
            //parsing
            //*******

        private:
            void ParseBegin();
            void ParseBeginFile();
            void ParseElementBeginName();
            void ParseElementHeader();
            void ParseElementData();
            void ParseElementEndName();
            void ParseAttributeName();
            void ParseAttributeValue();
            void ParseEndFile();
            void ParseEnd();
            void ParseComment();

            void (KXMLTokenizer::*fState)();
            void (KXMLTokenizer::*fInitialState)();
            void (KXMLTokenizer::*fFinalState)();

            int fNestedCommentCounter;

            //********
            //shipping
            //********

        private:
            string Trim( const string& aBuffer );
            string fBuffer;
            string fCommentBuffer;
            stack< string > fNames;
            KBeginParsingToken* fBeginParsing;
            KBeginFileToken* fBeginFile;
            KBeginElementToken* fBeginElement;
            KBeginAttributeToken* fBeginAttribute;
            KAttributeDataToken* fAttributeData;
            KEndAttributeToken* fEndAttribute;
            KMidElementToken* fMidElement;
            KElementDataToken* fElementData;
            KEndElementToken* fEndElement;
            KEndFileToken* fEndFile;
            KEndParsingToken* fEndParsing;
            KCommentToken* fComment;
            KErrorToken* fError;

            //**************
            //character sets
            //**************

        private:
            static const string fSpace;
            static const string fTab;
            static const string fNewLine;
            static const string fCarriageReturn;

            static const string fLowerCase;
            static const string fUpperCase;
            static const string fNumerals;

            static const string fLeftBraces;
            static const string fOperators;
            static const string fRightBraces;

            static const string fLeftAngle;
            static const string fRightAngle;
            static const string fLeftAngleSlash;
            static const string fRightSlashAngle;
            static const string fEqual;
            static const string fQuote;
            static const string fCommentStart;
            static const string fCommentEnd;

            static const string fParameterLeftBrace;
            static const string fParameterRightBrace;

            static const string fWhiteSpaceChars;
            static const string fNameStartChars;
            static const string fNameChars;
            static const string fValueChars;
    };

}

#endif
