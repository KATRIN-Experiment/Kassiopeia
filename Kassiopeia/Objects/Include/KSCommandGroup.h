#ifndef Kassiopeia_KSCommandGroup_h_
#define Kassiopeia_KSCommandGroup_h_

#include "KSCommand.h"

namespace Kassiopeia
{

    class KSCommandGroup :
        public KSCommand
    {
        public:
            KSCommandGroup();
            KSCommandGroup( const KSCommandGroup& aCopy );
            virtual ~KSCommandGroup();

        public:
            KSCommandGroup* Clone() const;

        public:
            void AddCommand( KSCommand* anCommand );
            void RemoveCommand( KSCommand* anCommand );

            KSCommand* CommandAt( unsigned int anIndex );
            const KSCommand* CommandAt( unsigned int anIndex ) const;
            unsigned int CommandCount() const;

        private:
            typedef std::vector< KSCommand* > CommandVector;
            typedef CommandVector::iterator CommandIt;
            typedef CommandVector::const_iterator CommandCIt;

            CommandVector fCommands;
    };

}

#endif
