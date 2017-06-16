#ifndef Kommon_KToken_hh_
#define Kommon_KToken_hh_

#include "KInitializationMessage.hh"

#include <string>
#include <sstream>

namespace katrin
{

    class KToken
    {
        public:
            KToken();
            KToken( const KToken& aToken );
            virtual ~KToken();

            virtual KToken* Clone() = 0;

        public:
            void SetValue( const std::string& aValue );
            const std::string& GetValue() const;

            template< typename XDataType >
            XDataType GetValue() const;

            void SetPath( const std::string& aPath );
            const std::string& GetPath() const;

            void SetFile( const std::string& aFile );
            const std::string& GetFile() const;

            void SetLine( const int& aLine );
            const int& GetLine() const;

            void SetColumn( const int& aColumn );
            const int& GetColumn() const;

        private:
            std::string fValue;

            std::string fPath;
            std::string fFile;
            int fLine;
            int fColumn;
    };

    template< typename XDataType >
    inline XDataType KToken::GetValue() const
    {
        std::istringstream Converter( fValue );
        XDataType Data;
        Converter >> Data;
        if (Converter.fail() || (Data != Data) )  // also check for NaN
        {
            std::string TypeName = KMessage::TypeName< XDataType >();
            initmsg( eWarning ) << "error processing token <" << fValue << "> with type <" << TypeName << ">, replaced with <" << Data << ">" << ret;
            initmsg( eWarning ) << "in path <" << fPath << "> in file <" << fFile << "> at line <" << fLine << "> at column <" << fColumn << ">" << eom;
        }
        return Data;
    }

    template<>
    inline bool KToken::GetValue< bool >() const
    {
        if ( fValue == std::string("0")
                || fValue == std::string("false") || fValue == std::string("False") || fValue == std::string("FALSE")
                || fValue == std::string("no") || fValue == std::string("No") || fValue == std::string("NO") )
        {
            return false;
        }
        return true;
    }

    template<>
    inline std::string KToken::GetValue< std::string >() const
    {
        return fValue;
    }

}

#endif
