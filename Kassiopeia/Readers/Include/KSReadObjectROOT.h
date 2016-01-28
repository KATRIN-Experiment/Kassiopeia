#ifndef _Kassiopeia_KSReadObjectROOT_h_
#define _Kassiopeia_KSReadObjectROOT_h_

#include "KSReadIterator.h"

#include "TFile.h"
#include "TTree.h"

namespace Kassiopeia
{
    class KSReadObjectROOT :
        public KSReadIterator,
        public KSBoolSet,
        public KSUCharSet,
        public KSCharSet,
        public KSUShortSet,
        public KSShortSet,
        public KSUIntSet,
        public KSIntSet,
        public KSULongSet,
        public KSLongSet,
        public KSFloatSet,
        public KSDoubleSet,
        public KSThreeVectorSet,
        public KSTwoVectorSet,
        public KSStringSet
    {
        public:
            using KSReadIterator::Add;
            using KSReadIterator::Get;
            using KSReadIterator::Exists;

        public:
            KSReadObjectROOT( TTree* aStructureTree, TTree* aPresenceTree, TTree* aDataTree );
            virtual ~KSReadObjectROOT();

        public:
            void operator++( int );
            void operator--( int );
            void operator<<( const unsigned int& aValue );

        public:
            bool Valid() const;
            unsigned int Index() const;
            bool operator<( const unsigned int& aValue ) const;
            bool operator<=( const unsigned int& aValue ) const;
            bool operator>( const unsigned int& aValue ) const;
            bool operator>=( const unsigned int& aValue ) const;
            bool operator==( const unsigned int& aValue ) const;
            bool operator!=( const unsigned int& aValue ) const;

        private:
            class Presence
            {
                public:
                    Presence( unsigned int anIndex, unsigned int aLength, unsigned int anEntry ) :
                            fIndex( anIndex ),
                            fLength( aLength ),
                            fEntry( anEntry )
                    {
                    }

                    unsigned int fIndex;
                    unsigned int fLength;
                    unsigned int fEntry;
            };

            vector< Presence > fPresences;
            bool fValid;
            unsigned int fIndex;
            TTree* fStructure;
            TTree* fPresence;
            TTree* fData;
    };

}

#endif
