#ifndef Kommon_KXMLTokenizer_hh_
#define Kommon_KXMLTokenizer_hh_

#include "KProcessor.hh"
#include "KFile.h"
#include "KTextFile.h"

#include <stack>
#include <vector>

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
            bool AtOneOf( const std::string& aSet );
            bool AtExactly( const std::string& aString );

            KTextFile* fFile;
            std::string fPath;
            std::string fName;
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
            std::string Trim( const std::string& aBuffer );
            std::string fBuffer;
            std::string fCommentBuffer;
            std::stack< std::string > fNames;
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
            static const std::string fSpace;
            static const std::string fTab;
            static const std::string fNewLine;
            static const std::string fCarriageReturn;

            static const std::string fLowerCase;
            static const std::string fUpperCase;
            static const std::string fNumerals;

            static const std::string fLeftBraces;
            static const std::string fOperators;
            static const std::string fRightBraces;

            static const std::string fLeftAngle;
            static const std::string fRightAngle;
            static const std::string fLeftAngleSlash;
            static const std::string fRightSlashAngle;
            static const std::string fEqual;
            static const std::string fQuote;
            static const std::string fCommentStart;
            static const std::string fCommentEnd;

            static const std::string fParameterLeftBrace;
            static const std::string fParameterRightBrace;

            static const std::string fWhiteSpaceChars;
            static const std::string fNameStartChars;
            static const std::string fNameChars;
            static const std::string fValueChars;
    };

}

#endif
