#ifndef KFILE_H_
#define KFILE_H_

#include <vector>
using std::vector;

#include <string>
using std::string;

#include "KAssert.h"

#include <cstdlib>

#define STRING(anArgument) #anArgument
#define AS_STRING(anArgument) STRING(anArgument)

#ifndef CONFIG_INSTALL_DIR
//KSTATICASSERT( false, CONFIG_INSTALL_DIR_was_not_defined )
#define CONFIG_DEFAULT_DIR "."
#else
#define CONFIG_DEFAULT_DIR AS_STRING( CONFIG_INSTALL_DIR )
#endif

#ifndef DATA_INSTALL_DIR
//KSTATICASSERT( false, DATA_INSTALL_DIR_was_not_defined )
#define DATA_DEFAULT_DIR "."
#else
#define DATA_DEFAULT_DIR AS_STRING( DATA_INSTALL_DIR )
#endif

#ifndef SCRATCH_INSTALL_DIR
//KSTATICASSERT( false, SCRATCH_INSTALL_DIR_was_not_defined )
#define SCRATCH_DEFAULT_DIR "."
#else
#define SCRATCH_DEFAULT_DIR AS_STRING( SCRATCH_INSTALL_DIR )
#endif

#ifndef OUTPUT_INSTALL_DIR
//KSTATICASSERT( false, OUTPUT_INSTALL_DIR_was_not_defined )
#define OUTPUT_DEFAULT_DIR "."
#else
#define OUTPUT_DEFAULT_DIR AS_STRING( OUTPUT_INSTALL_DIR )
#endif

#ifndef LOG_INSTALL_DIR
//KSTATICASSERT( false, LOG_INSTALL_DIR_was_not_defined )
#define LOG_DEFAULT_DIR "."
#else
#define LOG_DEFAULT_DIR AS_STRING(LOG_INSTALL_DIR)
#endif

namespace katrin
{

    class KFile
    {
        public:
            KFile();
            virtual ~KFile();

        public:
            void AddToPaths( const string& aPath );
            void SetDefaultPath( const string& aPath );
            void AddToBases( const string& aBase );
            void SetDefaultBase( const string& aBase );
            void AddToNames( const string& aName );
            const string& GetPath() const;
            const string& GetBase() const;
            const string& GetName() const;

        protected:
            vector< string > fPaths;
            string fDefaultPath;
            vector< string > fBases;
            string fDefaultBase;
            vector< string > fNames;
            string fResolvedPath;
            string fResolvedBase;
            string fResolvedName;

        public:
            typedef enum
            {
                eRead, eWrite, eAppend
            } Mode;

            typedef enum
            {
                eOpen, eClosed
            } State;

            bool Open( Mode aMode );
            bool IsOpen();

            bool Close();
            bool IsClosed();

        protected:
            virtual bool OpenFileSubclass( const string& aName, const Mode& aMode ) = 0;
            virtual bool CloseFileSubclass() = 0;
            State fState;

        protected:
            static const string fDirectoryMark;
            static const string fExtensionMark;
    };

}

#endif
