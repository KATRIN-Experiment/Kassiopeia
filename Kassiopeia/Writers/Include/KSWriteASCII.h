#ifndef Kassiopeia_KSWriteASCII_h_
#define Kassiopeia_KSWriteASCII_h_

#include "KSWriter.h"

#include "KFile.h"
using katrin::KFile;

#include "KRootFile.h"
using katrin::KRootFile;
using katrin::CreateOutputRootFile;

#include "KTextFile.h"
using katrin::KTextFile;
using katrin::CreateOutputTextFile;

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KTwoVector.hh"
using KGeoBag::KTwoVector;

#include <map>
using std::map;
using std::pair;

namespace Kassiopeia
{

	class KSWriteASCII :
	public KSComponentTemplate< KSWriteASCII, KSWriter >
	{
	private:
		class Data
		{
		public:
			Data( KSComponent* aComponent );
			Data( KSComponent* aComponent, KSWriteASCII* aWriter );
			~Data();

			void Start( const unsigned int& anIndex );
			void Fill();
			void MakeTitle( KSComponent* aComponent, int aTrack );

		private:
			string fLabel;
			string fType;
			KSWriteASCII* fWriter;
			unsigned int fIndex;
			unsigned int fLength;

			class Objekt
			{
			private:
				KSComponent* fComponent;
				string fType;
				int fPrecision;

			public:
				Objekt(KSComponent* aComponent, string aType, int Precision);
				~Objekt();
				string getValue();
			};

			vector< KSComponent* > fComponents;
			vector< Objekt* > fObjekts;
		};

		typedef map< KSComponent*, Data* > KSComponentMap;
		typedef KSComponentMap::iterator ComponentIt;
		typedef KSComponentMap::const_iterator ComponentCIt;
		typedef KSComponentMap::value_type ComponentEntry;

	public:
		KSWriteASCII();
		KSWriteASCII( const KSWriteASCII& aCopy );
		KSWriteASCII* Clone() const;
		virtual ~KSWriteASCII();

	public:
		void SetBase( const string& aBase );
		void SetPath( const string& aPath );
		void SetStepIteration( const unsigned int& aValue );
		void SetPrecision( const unsigned int& aValue );

		KTextFile* TextFile();
		int Precision() const;


	private:
		string fBase;
		string fPath;
		unsigned int fStepIteration;
		unsigned int fStepIterationIndex;
		unsigned int fPrecision;

	public:
		void ExecuteRun();
		void ExecuteEvent();
		void ExecuteTrack();
		void ExecuteStep();

		void AddRunComponent( KSComponent* aComponent );
		void RemoveRunComponent( KSComponent* aComponent );

		void AddEventComponent( KSComponent* aComponent );
		void RemoveEventComponent( KSComponent* aComponent );

		void AddTrackComponent( KSComponent* aComponent );
		void RemoveTrackComponent( KSComponent* aComponent );

		void AddStepComponent( KSComponent* aComponent );
		void RemoveStepComponent( KSComponent* aComponent );

	protected:
		virtual void InitializeComponent();
		virtual void DeinitializeComponent();

	private:

		KTextFile* fTextFile;

		string fKey;
		KSComponentMap fRunComponents;
		KSComponentMap fActiveRunComponents;
		unsigned int fRunIndex;
		unsigned int fRunFirstEventIndex;
		unsigned int fRunLastEventIndex;
		unsigned int fRunFirstTrackIndex;
		unsigned int fRunLastTrackIndex;
		unsigned int fRunFirstStepIndex;
		unsigned int fRunLastStepIndex;

		KSComponentMap fEventComponents;
		KSComponentMap fActiveEventComponents;
		unsigned int fEventIndex;
		unsigned int fEventFirstTrackIndex;
		unsigned int fEventLastTrackIndex;
		unsigned int fEventFirstStepIndex;
		unsigned int fEventLastStepIndex;

		KSComponentMap fTrackComponents;
		KSComponentMap fActiveTrackComponents;
		unsigned int fTrackIndex;
		unsigned int fTrackFirstStepIndex;
		unsigned int fTrackLastStepIndex;

		bool fStepComponent;
		KSComponentMap fStepComponents;
		KSComponentMap fActiveStepComponents;
		unsigned int fStepIndex;

		static const Int_t fBufferSize;
		static const Int_t fSplitLevel;
		static const string fLabel;

	};

	inline void KSWriteASCII::SetBase( const string& aBase )
	{
		fBase = aBase;
		return;
	}
	inline void KSWriteASCII::SetPath( const string& aPath )
	{
		fPath = aPath;
		return;
	}
	inline void KSWriteASCII::SetStepIteration( const unsigned int& aValue )
	{
		fStepIteration = aValue;
		return;
	}

	inline void KSWriteASCII::SetPrecision( const unsigned int& aValue )
	{
		fPrecision = aValue;
		return;
	}

	inline KTextFile* KSWriteASCII::TextFile( )
	{
		
		return fTextFile;
	}

	inline int KSWriteASCII::Precision( ) const
	{
		return fPrecision;
	}

}


#endif
