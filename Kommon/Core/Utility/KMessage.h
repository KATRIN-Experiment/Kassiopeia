#ifndef KMESSAGE_H_
#define KMESSAGE_H_

#include <vector>
using std::vector;

#include <map>
using std::map;

#include <utility>
using std::pair;

#include <string>
using std::string;

#include <sstream>
using std::stringstream;

#include <ostream>
using std::ostream;

#include <iomanip>
using std::setprecision;
using std::fixed;
using std::scientific;

#include <cstdlib>
#include <cxxabi.h>  // needed to convert typename to string

namespace katrin
{

    class KMessageNewline
    {
    };

    class KMessageOverline
    {
    };

    class KMessageNewlineEnd
    {
    };

    class KMessageOverlineEnd
    {
    };

    typedef std::ios_base::fmtflags KMessageFormat;
    typedef std::streamsize KMessagePrecision;
    typedef int KMessageSeverity;
    static const KMessageSeverity eError = 0;
    static const KMessageSeverity eWarning = 1;
    static const KMessageSeverity eNormal = 2;
    static const KMessageSeverity eDebug = 3;
    static const KMessageNewline ret = KMessageNewline();
    static const KMessageOverline rret = KMessageOverline();
    static const KMessageNewlineEnd eom = KMessageNewlineEnd();
    static const KMessageOverlineEnd reom = KMessageOverlineEnd();

    class KMessage
    {
        public:
            KMessage( const string& aKey, const string& aDescription, const string& aPrefix, const string& aSuffix );
            virtual ~KMessage();

        private:
            KMessage();
            KMessage( const KMessage& );

            //**************
            //identification
            //**************

        public:
            const string& GetKey();
            void SetKey( const string& aKey );

        protected:
            string fKey;

            //*********
            //interface
            //*********

        public:
            /**
             * Helper function to convert typename to human-readable string, see: http://stackoverflow.com/a/19123821
             */
            template< typename XDataType >
            static string TypeName();

        public:
            KMessage& operator()( const KMessageSeverity& );

            template< class XPrintable >
            KMessage& operator<<( const XPrintable& aFragment );
            KMessage& operator<<( const KMessageNewline& );
            KMessage& operator<<( const KMessageOverline& );
            KMessage& operator<<( const KMessageNewlineEnd& );
            KMessage& operator<<( const KMessageOverlineEnd& );

        private:
            void SetSeverity( const KMessageSeverity& aSeverity );
            void Flush();
            void Shutdown();

        protected:
            string fSystemDescription;
            string fSystemPrefix;
            string fSystemSuffix;

            string fErrorColorPrefix;
            string fErrorColorSuffix;
            string fErrorDescription;

            string fWarningColorPrefix;
            string fWarningColorSuffix;
            string fWarningDescription;

            string fNormalColorPrefix;
            string fNormalColorSuffix;
            string fNormalDescription;

            string fDebugColorPrefix;
            string fDebugColorSuffix;
            string fDebugDescription;

            string fDefaultColorPrefix;
            string fDefaultColorSuffix;
            string fDefaultDescription;

        private:
            KMessageSeverity fSeverity;

            string KMessage::*fColorPrefix;
            string KMessage::*fDescription;
            string KMessage::*fColorSuffix;

            stringstream fMessageLine;
            vector< pair< string, char > > fMessageLines;

            //********
            //settings
            //********

        public:
            void SetFormat( const KMessageFormat& aFormat );
            void SetPrecision( const KMessagePrecision& aPrecision );
            void SetTerminalVerbosity( const KMessageSeverity& aVerbosity );
            void SetTerminalStream( ostream* aTerminalStream );
            void SetLogVerbosity( const KMessageSeverity& aVerbosity );
            void SetLogStream( ostream* aLogStream );

        private:
            KMessageSeverity fTerminalVerbosity;
            ostream* fTerminalStream;
            KMessageSeverity fLogVerbosity;
            ostream* fLogStream;

    };

    template< typename XDataType >
    string KMessage::TypeName()
    {
        int StatusFlag;
        string TypeName = typeid( XDataType ).name();
        char *DemangledName = abi::__cxa_demangle( TypeName.c_str(), NULL, NULL, &StatusFlag );
        if ( StatusFlag == 0 )
        {
            TypeName = string( DemangledName );
            free( DemangledName );
        }
        return TypeName;
    }

    inline KMessage& KMessage::operator()( const KMessageSeverity& aSeverity )
    {
        SetSeverity( aSeverity );
        return *this;
    }

    template< class XPrintable >
    KMessage& KMessage::operator<<( const XPrintable& aFragment )
    {
        fMessageLine << aFragment;
        return *this;
    }
    inline KMessage& KMessage::operator<<( const KMessageNewline& )
    {
        fMessageLines.push_back( pair< string, char >( fMessageLine.str(), '\n' ) );
        fMessageLine.clear();
        fMessageLine.str( "" );
        return *this;
    }
    inline KMessage& KMessage::operator<<( const KMessageOverline& )
    {
        fMessageLines.push_back( pair< string, char >( fMessageLine.str(), '\r' ) );
        fMessageLine.clear();
        fMessageLine.str( "" );
        return *this;
    }
    inline KMessage& KMessage::operator<<( const KMessageNewlineEnd& )
    {
        fMessageLines.push_back( pair< string, char >( fMessageLine.str(), '\n' ) );
        fMessageLine.clear();
        fMessageLine.str( "" );
        Flush();
        return *this;
    }
    inline KMessage& KMessage::operator<<( const KMessageOverlineEnd& )
    {
        fMessageLines.push_back( pair< string, char >( fMessageLine.str(), '\r' ) );
        fMessageLine.clear();
        fMessageLine.str( "" );
        Flush();
        return *this;
    }
}

#include "KSingleton.h"

namespace katrin
{

    class KMessageTable :
        public KSingleton< KMessageTable >
    {

        public:
            friend class KSingleton< KMessageTable > ;

        private:
            KMessageTable();
            ~KMessageTable();

        public:
            void Add( KMessage* aMessage );
            KMessage* Get( const string& aKey );
            void Remove( KMessage* aMessage );

            void SetFormat( const KMessageFormat& aFormat );
            const KMessageFormat& GetFormat();

            void SetPrecision( const KMessagePrecision& aPrecision );
            const KMessagePrecision& GetPrecision();

            void SetTerminalVerbosity( const KMessageSeverity& aVerbosity );
            const KMessageSeverity& GetTerminalVerbosity();

            void SetTerminalStream( ostream* aTerminalStream );
            ostream* GetTerminalStream();

            void SetLogVerbosity( const KMessageSeverity& aVerbosity );
            const KMessageSeverity& GetLogVerbosity();

            void SetLogStream( ostream* aLogStream );
            ostream* GetLogStream();

        private:
            typedef map< string, KMessage* > MessageMap;
            typedef MessageMap::value_type MessageEntry;
            typedef MessageMap::iterator MessageIt;
            typedef MessageMap::const_iterator MessageCIt;

            MessageMap fMessageMap;

            KMessageFormat fFormat;
            KMessagePrecision fPrecision;
            KMessageSeverity fTerminalVerbosity;
            ostream* fTerminalStream;
            KMessageSeverity fLogVerbosity;
            ostream* fLogStream;
    };

}

#include "KInitializer.h"

#define KMESSAGE_DECLARE( xNAMESPACE, xNAME )\
namespace xNAMESPACE\
{\
    class __attribute__((__may_alias__)) KMessage_ ## xNAME :\
        public katrin::KMessage\
    {\
        public:\
            KMessage_ ## xNAME();\
            virtual ~KMessage_ ## xNAME();\
    };\
\
    using katrin::eDebug;\
    using katrin::eNormal;\
    using katrin::eWarning;\
    using katrin::eError;\
\
    using katrin::ret;\
    using katrin::rret;\
    using katrin::eom;\
    using katrin::reom;\
\
    extern KMessage_ ## xNAME& xNAME;\
    static katrin::KInitializer< KMessage_ ## xNAME > xNAME ## _initializer;\
}

#define KMESSAGE_DEFINE( xNAMESPACE, xNAME, xKEY, xLABEL )\
namespace xNAMESPACE\
{\
    KMessage_ ## xNAME::KMessage_ ## xNAME() :\
        katrin::KMessage( #xKEY, #xLABEL, "", "" )\
    {\
    }\
    KMessage_ ## xNAME::~KMessage_ ## xNAME()\
    {\
    }\
\
    KMessage_ ## xNAME& xNAME = *((KMessage_ ## xNAME*) (katrin::KInitializer< KMessage_ ## xNAME >::fData));\
}

#define KMESSAGE_DEFINE_FULL( xNAMESPACE, xNAME, xKEY, xLABEL, xPREFIX, xSUFFIX )\
namespace xNAMESPACE\
{\
    KMessage_ ## xNAME::KMessage_ ## xNAME() :\
        katrin::KMessage( #xKEY, #xLABEL, #xPREFIX, #xSUFFIX )\
    {\
    }\
    KMessage_ ## xNAME::~KMessage_ ## xNAME()\
    {\
    }\
\
    KMessage_ ## xNAME& xNAME = *((KMessage_ ## xNAME*) (katrin::KInitializer< KMessage_ ## xNAME >::fData));\
}

#endif
