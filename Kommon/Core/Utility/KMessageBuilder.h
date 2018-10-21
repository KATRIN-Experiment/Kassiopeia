#ifndef KMESSAGEBUILDER_H_
#define KMESSAGEBUILDER_H_

#include "KMessage.h"
#include "KComplexElement.hh"
#include "KContainer.hh"
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
            if( aContainer->AsReference< std::string >() == std::string( "error" ) )
            {
                fObject->fTerminalVerbosity = eError;
                return true;
            }
            if( aContainer->AsReference< std::string >() == std::string( "warning" ) )
            {
                fObject->fTerminalVerbosity = eWarning;
                return true;
            }
            if( aContainer->AsReference< std::string >() == std::string( "normal" ) )
            {
                fObject->fTerminalVerbosity = eNormal;
                return true;
            }
            if( aContainer->AsReference< std::string >() == std::string( "info" ) )
            {
                fObject->fTerminalVerbosity = eInfo;
                return true;
            }
            if( aContainer->AsReference< std::string >() == std::string( "debug" ) )
            {
                fObject->fTerminalVerbosity = eDebug;
                return true;
            }
            return false;
        }
        if( aContainer->GetName() == "log" )
        {
            if( aContainer->AsReference< std::string >() == std::string( "error" ) )
            {
                fObject->fLogVerbosity = eError;
                return true;
            }
            if( aContainer->AsReference< std::string >() == std::string( "warning" ) )
            {
                fObject->fLogVerbosity = eWarning;
                return true;
            }
            if( aContainer->AsReference< std::string >() == std::string( "normal" ) )
            {
                fObject->fLogVerbosity = eNormal;
                return true;
            }
            if( aContainer->AsReference< std::string >() == std::string( "info" ) )
            {
                fObject->fLogVerbosity = eInfo;
                return true;
            }
            if( aContainer->AsReference< std::string >() == std::string( "debug" ) )
            {
                fObject->fLogVerbosity = eDebug;
                return true;
            }
            return false;
        }
        if( aContainer->GetName() == "format" )
        {
            if( aContainer->AsReference< std::string >() == std::string( "fixed" ) )
            {
                fObject->fFormat = std::ios_base::fixed;
                return true;
            }
            if( aContainer->AsReference< std::string >() == std::string( "scientific" ) )
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
        fObject = &KMessageTable::GetInstance();
        return true;
    }

    template< >
    inline bool KMessageTableBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "terminal" )
        {
            if( anAttribute->AsReference< std::string >() == std::string( "error" ) )
            {
                fObject->SetTerminalVerbosity( eError );
                return true;
            }
            if( anAttribute->AsReference< std::string >() == std::string( "warning" ) )
            {
                fObject->SetTerminalVerbosity( eWarning );
                return true;
            }
            if( anAttribute->AsReference< std::string >() == std::string( "normal" ) )
            {
                fObject->SetTerminalVerbosity( eNormal );
                return true;
            }
            if( anAttribute->AsReference< std::string >() == std::string( "info" ) )
            {
                fObject->SetTerminalVerbosity( eInfo );
                return true;
            }
            if( anAttribute->AsReference< std::string >() == std::string( "debug" ) )
            {
                fObject->SetTerminalVerbosity( eDebug );
                return true;
            }
            return false;
        }
        if( anAttribute->GetName() == "log" )
        {
            if( anAttribute->AsReference< std::string >() == std::string( "error" ) )
            {
                fObject->SetLogVerbosity( eError );
                return true;
            }
            if( anAttribute->AsReference< std::string >() == std::string( "warning" ) )
            {
                fObject->SetLogVerbosity( eWarning );
                return true;
            }
            if( anAttribute->AsReference< std::string >() == std::string( "normal" ) )
            {
                fObject->SetLogVerbosity( eNormal );
                return true;
            }
            if( anAttribute->AsReference< std::string >() == std::string( "info" ) )
            {
                fObject->SetLogVerbosity( eInfo );
                return true;
            }
            if( anAttribute->AsReference< std::string >() == std::string( "debug" ) )
            {
                fObject->SetLogVerbosity( eDebug );
                return true;
            }
            return false;
        }
        if( anAttribute->GetName() == "format" )
        {
            if( anAttribute->AsReference< std::string >() == std::string( "fixed" ) )
            {
                fObject->SetFormat( std::ios_base::fixed );
                return true;
            }
            if( anAttribute->AsReference< std::string >() == std::string( "scientific" ) )
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
            KTextFile* tFile = NULL;
            anElement->ReleaseTo(tFile);

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
            if ( tMessageData->fKey == std::string("all"))
            {
                KMessageTable::GetInstance().SetTerminalVerbosity( tMessageData->fTerminalVerbosity );
                KMessageTable::GetInstance().SetLogVerbosity( tMessageData->fLogVerbosity );
                KMessageTable::GetInstance().SetFormat( tMessageData->fFormat );
                KMessageTable::GetInstance().SetPrecision( tMessageData->fPrecision );
                return true;
            }
            KMessage* tMessage = KMessageTable::GetInstance().Get( tMessageData->fKey );
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
