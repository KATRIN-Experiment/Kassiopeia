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
    class KMessage_ ## xNAME :\
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

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

#include "KComplexElement.hh"
#include "KTextFile.h"

namespace katrin
{

    class KMessageData
    {
        public:
            KMessageData();
            ~KMessageData();

        public:
            string fKey;
            KMessageFormat fFormat;
            KMessagePrecision fPrecision;
            KMessageSeverity fTerminalVerbosity;
            KMessageSeverity fLogVerbosity;
    };

    typedef KComplexElement< KMessageData > KMessageDataBuilder;

    template< >
    inline bool KMessageDataBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "key" )
        {
            aContainer->CopyTo( fObject->fKey );
            return true;
        }
        if( aContainer->GetName() == "terminal" )
        {
            if( aContainer->AsReference< string >() == string( "error" ) )
            {
                fObject->fTerminalVerbosity = eError;
                return true;
            }
            if( aContainer->AsReference< string >() == string( "warning" ) )
            {
                fObject->fTerminalVerbosity = eWarning;
                return true;
            }
            if( aContainer->AsReference< string >() == string( "normal" ) )
            {
                fObject->fTerminalVerbosity = eNormal;
                return true;
            }
            if( aContainer->AsReference< string >() == string( "debug" ) )
            {
                fObject->fTerminalVerbosity = eDebug;
                return true;
            }
            return false;
        }
        if( aContainer->GetName() == "log" )
        {
            if( aContainer->AsReference< string >() == string( "error" ) )
            {
                fObject->fLogVerbosity = eError;
                return true;
            }
            if( aContainer->AsReference< string >() == string( "warning" ) )
            {
                fObject->fLogVerbosity = eWarning;
                return true;
            }
            if( aContainer->AsReference< string >() == string( "normal" ) )
            {
                fObject->fLogVerbosity = eNormal;
                return true;
            }
            if( aContainer->AsReference< string >() == string( "debug" ) )
            {
                fObject->fLogVerbosity = eDebug;
                return true;
            }
            return false;
        }
        if( aContainer->GetName() == "format" )
        {
            if( aContainer->AsReference< string >() == string( "fixed" ) )
            {
                fObject->fFormat = std::ios_base::fixed;
                return true;
            }
            if( aContainer->AsReference< string >() == string( "scientific" ) )
            {
                fObject->fFormat = std::ios_base::scientific;
                return true;
            }
            return false;
        }
        if( aContainer->GetName() == "precision" )
        {
            fObject->fPrecision = aContainer->AsReference< KMessagePrecision >();
            return true;
        }
        return false;
    }


    typedef KComplexElement< KMessageTable > KMessageTableBuilder;

    template< >
    inline bool KMessageTableBuilder::Begin()
    {
        fObject = KMessageTable::GetInstance();
        return true;
    }

    template< >
    inline bool KMessageTableBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "terminal" )
        {
            if( anAttribute->AsReference< string >() == string( "error" ) )
            {
                fObject->SetTerminalVerbosity( eError );
                return true;
            }
            if( anAttribute->AsReference< string >() == string( "warning" ) )
            {
                fObject->SetTerminalVerbosity( eWarning );
                return true;
            }
            if( anAttribute->AsReference< string >() == string( "normal" ) )
            {
                fObject->SetTerminalVerbosity( eNormal );
                return true;
            }
            if( anAttribute->AsReference< string >() == string( "debug" ) )
            {
                fObject->SetTerminalVerbosity( eDebug );
                return true;
            }
            return false;
        }
        if( anAttribute->GetName() == "log" )
        {
            if( anAttribute->AsReference< string >() == string( "error" ) )
            {
                fObject->SetLogVerbosity( eError );
                return true;
            }
            if( anAttribute->AsReference< string >() == string( "warning" ) )
            {
                fObject->SetLogVerbosity( eWarning );
                return true;
            }
            if( anAttribute->AsReference< string >() == string( "normal" ) )
            {
                fObject->SetLogVerbosity( eNormal );
                return true;
            }
            if( anAttribute->AsReference< string >() == string( "debug" ) )
            {
                fObject->SetLogVerbosity( eDebug );
                return true;
            }
            return false;
        }
        if( anAttribute->GetName() == "format" )
        {
            if( anAttribute->AsReference< string >() == string( "fixed" ) )
            {
                fObject->SetFormat( std::ios_base::fixed );
                return true;
            }
            if( anAttribute->AsReference< string >() == string( "scientific" ) )
            {
                fObject->SetFormat( std::ios_base::scientific );
                return true;
            }
            return false;
        }
        if( anAttribute->GetName() == "precision" )
        {
            fObject->SetPrecision( anAttribute->AsReference< KMessagePrecision >() );
            return true;
        }
        return false;
    }

    template< >
    inline bool KMessageTableBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "file" )
        {
            KTextFile* tFile;
            anElement->ReleaseTo( tFile );

            tFile->SetDefaultPath( LOG_DEFAULT_DIR );
            tFile->SetDefaultBase( "KasperLog.txt" );
            tFile->Open( KFile::eWrite );
            if( tFile->IsOpen() == true )
            {
                fObject->SetLogStream( tFile->File() );
            }
            else
            {
                KMessage tMessage( "k_common", "KOMMON", "", "" );
                tMessage( eWarning ) << "could not open logfile" << eom;
            }
            return true;
        }
        if( anElement->GetName() == "message" )
        {
            KMessageData* tMessageData = anElement->AsPointer< KMessageData >();
            if ( tMessageData->fKey == string("all"))
            {
            	KMessageTable::GetInstance()->SetTerminalVerbosity( tMessageData->fTerminalVerbosity );
            	KMessageTable::GetInstance()->SetLogVerbosity( tMessageData->fLogVerbosity );
            	KMessageTable::GetInstance()->SetFormat( tMessageData->fFormat );
            	KMessageTable::GetInstance()->SetPrecision( tMessageData->fPrecision );
            	return true;
            }
            KMessage* tMessage = KMessageTable::GetInstance()->Get( tMessageData->fKey );
            if( tMessage != NULL )
            {
                tMessage->SetTerminalVerbosity( tMessageData->fTerminalVerbosity );
                tMessage->SetLogVerbosity( tMessageData->fLogVerbosity );
                tMessage->SetFormat( tMessageData->fFormat );
                tMessage->SetPrecision( tMessageData->fPrecision );
            }
            else
            {
                KMessage tMessage( "k_common", "KOMMON", "", "" );
                tMessage( eWarning ) << "no message registered with key <" << tMessageData->fKey << ">" << eom;
            }
            return true;
        }
        return false;
    }

    template< >
    inline bool KMessageTableBuilder::End()
    {
        fObject = NULL;
        return true;
    }

}

#endif
