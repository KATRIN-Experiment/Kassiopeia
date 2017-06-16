#ifndef KMESSAGE_H_
#define KMESSAGE_H_

#include <vector>
#include <map>
#include <utility>
#include <string>
#include <sstream>
#include <ostream>
#include <iomanip>
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
            KMessage( const std::string& aKey, const std::string& aDescription, const std::string& aPrefix, const std::string& aSuffix );
            virtual ~KMessage();

        private:
            KMessage();
            KMessage( const KMessage& );

            //**************
            //identification
            //**************

        public:
            const std::string& GetKey();
            void SetKey( const std::string& aKey );

        protected:
            std::string fKey;

            //*********
            //interface
            //*********

        public:
            /**
             * Helper function to convert typename to human-readable string, see: http://stackoverflow.com/a/19123821
             */
            template< typename XDataType >
            static std::string TypeName();

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
            std::string fSystemDescription;
            std::string fSystemPrefix;
            std::string fSystemSuffix;

            std::string fErrorColorPrefix;
            std::string fErrorColorSuffix;
            std::string fErrorDescription;

            std::string fWarningColorPrefix;
            std::string fWarningColorSuffix;
            std::string fWarningDescription;

            std::string fNormalColorPrefix;
            std::string fNormalColorSuffix;
            std::string fNormalDescription;

            std::string fDebugColorPrefix;
            std::string fDebugColorSuffix;
            std::string fDebugDescription;

            std::string fDefaultColorPrefix;
            std::string fDefaultColorSuffix;
            std::string fDefaultDescription;

        private:
            KMessageSeverity fSeverity;

            std::string KMessage::*fColorPrefix;
            std::string KMessage::*fDescription;
            std::string KMessage::*fColorSuffix;

            std::stringstream fMessageLine;
            std::vector< std::pair< std::string, char > > fMessageLines;

            //********
            //settings
            //********

        public:
            void SetFormat( const KMessageFormat& aFormat );
            void SetPrecision( const KMessagePrecision& aPrecision );
            void SetTerminalVerbosity( const KMessageSeverity& aVerbosity );
            void SetTerminalStream( std::ostream* aTerminalStream );
            void SetLogVerbosity( const KMessageSeverity& aVerbosity );
            void SetLogStream( std::ostream* aLogStream );

        private:
            KMessageSeverity fTerminalVerbosity;
            std::ostream* fTerminalStream;
            KMessageSeverity fLogVerbosity;
            std::ostream* fLogStream;

    };

    template< typename XDataType >
    std::string KMessage::TypeName()
    {
        int StatusFlag;
        std::string TypeName = typeid( XDataType ).name();
        char *DemangledName = abi::__cxa_demangle( TypeName.c_str(), NULL, NULL, &StatusFlag );
        if ( StatusFlag == 0 )
        {
            TypeName = std::string( DemangledName );
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
        fMessageLines.push_back( std::pair< std::string, char >( fMessageLine.str(), '\n' ) );
        fMessageLine.clear();
        fMessageLine.str( "" );
        return *this;
    }
    inline KMessage& KMessage::operator<<( const KMessageOverline& )
    {
        fMessageLines.push_back( std::pair< std::string, char >( fMessageLine.str(), '\r' ) );
        fMessageLine.clear();
        fMessageLine.str( "" );
        return *this;
    }
    inline KMessage& KMessage::operator<<( const KMessageNewlineEnd& )
    {
        fMessageLines.push_back( std::pair< std::string, char >( fMessageLine.str(), '\n' ) );
        fMessageLine.clear();
        fMessageLine.str( "" );
        Flush();
        return *this;
    }
    inline KMessage& KMessage::operator<<( const KMessageOverlineEnd& )
    {
        fMessageLines.push_back( std::pair< std::string, char >( fMessageLine.str(), '\r' ) );
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
            KMessage* Get( const std::string& aKey );
            void Remove( KMessage* aMessage );

            void SetFormat( const KMessageFormat& aFormat );
            const KMessageFormat& GetFormat();

            void SetPrecision( const KMessagePrecision& aPrecision );
            const KMessagePrecision& GetPrecision();

            void SetTerminalVerbosity( const KMessageSeverity& aVerbosity );
            const KMessageSeverity& GetTerminalVerbosity();

            void SetTerminalStream( std::ostream* aTerminalStream );
            std::ostream* GetTerminalStream();

            void SetLogVerbosity( const KMessageSeverity& aVerbosity );
            const KMessageSeverity& GetLogVerbosity();

            void SetLogStream( std::ostream* aLogStream );
            std::ostream* GetLogStream();

        private:
            typedef std::map< std::string, KMessage* > MessageMap;
            typedef MessageMap::value_type MessageEntry;
            typedef MessageMap::iterator MessageIt;
            typedef MessageMap::const_iterator MessageCIt;

            MessageMap fMessageMap;

            KMessageFormat fFormat;
            KMessagePrecision fPrecision;
            KMessageSeverity fTerminalVerbosity;
            std::ostream* fTerminalStream;
            KMessageSeverity fLogVerbosity;
            std::ostream* fLogStream;
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
