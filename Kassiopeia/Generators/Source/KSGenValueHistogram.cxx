#include "KSGenValueHistogram.h"

#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{

    KSGenValueHistogram::KSGenValueHistogram() :
            fBase( "" ),
            fPath( "" ),
            fHistogram( "" ),
            fFormula( "" ),
            fRootFile( NULL ),
            fValueHistogram( NULL ),
            fValueFunction( NULL )
    {
    }
    KSGenValueHistogram::KSGenValueHistogram( const KSGenValueHistogram& aCopy ) :
            KSComponent(),
            fBase( aCopy.fBase ),
            fPath( aCopy.fPath ),
            fHistogram( aCopy.fHistogram ),
            fFormula( aCopy.fFormula ),
            fRootFile( NULL ),
            fValueHistogram( NULL ),
            fValueFunction( NULL )
    {
    }
    KSGenValueHistogram* KSGenValueHistogram::Clone() const
    {
        return new KSGenValueHistogram( *this );
    }
    KSGenValueHistogram::~KSGenValueHistogram()
    {
    }

    void KSGenValueHistogram::DiceValue( vector< double >& aDicedValues )
    {
        double tValue;

        tValue = fValueHistogram->GetRandom();
        genmsg_debug( "histogram generator <" << GetName() << "> diced value <" << tValue << "> from histogram <" << fHistogram << ">" << eom);
        if ( fValueFunction != NULL )
        {
            tValue = fValueFunction->Eval( tValue );
            genmsg_debug( "histogram generator <" << GetName() << "> modified diced value to <" << tValue << "> via formula <" << fFormula << ">" << eom );
        }
        aDicedValues.push_back( tValue );

        return;
    }

    void KSGenValueHistogram::InitializeComponent()
    {
        fRootFile = KRootFile::CreateDataRootFile( fBase );
        if( ! fPath.empty() )
        {
            fRootFile->AddToPaths( fPath );
        }
        if( fRootFile->Open( KFile::eRead ) == false )
        {
            genmsg( eError ) << "histogram generator <" << GetName() << "> could not open file <" << fBase << "> at path <" << fPath << ">" << eom;
        }

        fValueHistogram = static_cast< TH1* >( fRootFile->File()->Get( fHistogram.c_str() ) );
        fValueHistogram->SetDirectory(0);

        if ( fValueHistogram == NULL )
        {
            genmsg( eError ) << "histogram generator <" << GetName() << "> could not find ROOT histogram <" << fHistogram << ">" << eom;
        }

        if ( ! fFormula.empty() )
        {
            double tValueMin = fValueHistogram->GetXaxis()->GetBinCenter( 1 ); // 0 is underflow
            double tValueMax = fValueHistogram->GetXaxis()->GetBinCenter( fValueHistogram->GetNbinsX() );  // nbins+1 is overflow
            fValueFunction = new TF1( "function", fFormula.c_str(), tValueMin, tValueMax );
        }

        fRootFile->Close();

        return;
    }
    void KSGenValueHistogram::DeinitializeComponent()
    {
        if ( fRootFile != NULL )
            delete fRootFile;
        fRootFile = NULL;

        if ( fValueHistogram != NULL )
            delete fValueHistogram;
        fValueHistogram = NULL;

        if ( fValueFunction != NULL )
            delete fValueFunction;
        fValueFunction = NULL;

        return;
    }

}
