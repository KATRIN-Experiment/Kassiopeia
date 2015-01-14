#include "KSMainMessage.h"
#include "KRootFile.h"

#include "TFile.h"
#include "TObjString.h"
#include "TTree.h"
#include "TObjArray.h"
#include "TBranch.h"
#include "TLeaf.h"

#include <string>
#include <iostream>
#include <map>

using namespace std;
using namespace katrin;
using namespace Kassiopeia;

typedef vector< pair< string, TTree* > > DataTreeVector;

struct Trees {
    TTree *tRunTree;
    TTree *tEventTree;
    TTree *tTrackTree;
    TTree *tStepTree;

    TTree *tRunKeysTree;
    TTree *tEventKeysTree;
    TTree *tTrackKeysTree;
    TTree *tStepKeysTree;
};

struct Indices {
    unsigned int tRunIndex;
    unsigned int tRunFirstEvent;
    unsigned int tRunLastEvent;
    unsigned int tRunFirstTrack;
    unsigned int tRunLastTrack;
    unsigned int tRunFirstStep;
    unsigned int tRunLastStep;
    unsigned int tEventIndex;
    unsigned int tEventFirstTrack;
    unsigned int tEventLastTrack;
    unsigned int tEventFirstStep;
    unsigned int tEventLastStep;
    unsigned int tTrackIndex;
    unsigned int tTrackFirstStep;
    unsigned int tTrackLastStep;
    unsigned int tStepIndex;
};

struct DataMap {

    DataMap() :
        tStructureLabelPointer( &tStructureLabel ),
        tStructureLabelHandle( &tStructureLabelPointer ),
        tStructureTypePointer( &tStructureType ),
        tStructureTypeHandle( &tStructureTypePointer ),
        tPresenceIndex( 0 ),
        tPresenceIndexPointer( &tPresenceIndex ),
        tPresenceLength( 0 ),
        tPresenceLengthPointer( &tPresenceLength )
    {
    }

    string tStructureLabel;
    string *tStructureLabelPointer;
    string **tStructureLabelHandle;
    string tStructureType;
    string *tStructureTypePointer;
    string **tStructureTypeHandle;
    unsigned int tPresenceIndex;
    unsigned int *tPresenceIndexPointer;
    unsigned int tPresenceLength;
    unsigned int *tPresenceLengthPointer;
    map< string, Bool_t > tBoolMap;
    map< string, UChar_t > tUCharMap;
    map< string, Char_t > tCharMap;
    map< string, UShort_t > tUShortMap;
    map< string, Short_t > tShortMap;
    map< string, UInt_t > tUIntMap;
    map< string, Int_t > tIntMap;
    map< string, ULong64_t > tULongMap;
    map< string, Long64_t > tLongMap;
    map< string, Float_t > tFloatMap;
    map< string, Double_t > tDoubleMap;
    map< string, string > tStringMap;
    map< string, string* > tStringPointerMap;
    map< string, Double_t > tTwoVectorXMap;
    map< string, Double_t > tTwoVectorYMap;
    map< string, Double_t > tThreeVectorXMap;
    map< string, Double_t > tThreeVectorYMap;
    map< string, Double_t > tThreeVectorZMap;
};

//function declarations
TTree* GetTree( KRootFile *tInputRootFile, TString tName );
void CheckLabel( KRootFile *tInputRootFile );
void GetIndicesBranchAddresses( Trees *tTrees, Indices *tIndices );
void GetDataTrees( KRootFile *tInputFile, TTree *tKeyTree, DataTreeVector *tDataTreeVector, bool first );
void GetDataBranches( TTree *tStructureTree, TTree *tPresenceTree, TTree *tDataTree, DataMap *tDataMap );
void WriteIndexTrees( KRootFile *tRootFile, Trees *tTrees, Indices *tIndices, const int tBufferSize, const int tSplitSize );
void WriteDataTrees( KRootFile *tRootFile, DataTreeVector *tOutputDataTreeVector, DataTreeVector *tDataTreeVector );
void WriteDataBranches( TTree *tStructureTree, TTree *tPresenceTree, TTree *tDataTree, DataMap *tDataMap, const int tBufferSize, const int tSplitSize );
void FillDataStructureBranch( TTree *tOutputStructureTree, TTree *tInputStructureTree );
void GetWriteFillKeyTrees( KRootFile *tRootFile, Trees *tInputTrees, Trees *tOutputTrees, const int tBufferSize, const int tSplitSize );
void FillIndexTrees( Trees *tOutputTrees, Trees *tInputTrees, Indices *tIndices );
void FillDataBranches( TTree *tOutputDataTree, TTree *tInputDataTree );
void FillDataPresenceBranch( TTree *tOutputDataPresenceTree, TTree *tInputDataPresenceTree, DataMap *tDataMap );


int main( int argc, char** argv )
{
    KMessageTable::GetInstance()->SetTerminalVerbosity( eNormal );
    KMessageTable::GetInstance()->SetLogVerbosity( eNormal );

    const int tBufferSize = 64000;
    const int tSplitSize = 99;

    if( argc < 4 )
    {
        cout << "usage: ./ROOTFileMerge <input_file_1> <input_file_2> [<input_file_3> <...>] <output_file>" << endl;
        exit( -1 );
    }
    mainmsg( eNormal ) << "Analyzing " << argc - 2 << " files" << eom;

    Trees *tInputTrees = new Trees();
    Trees *tOutputTrees = new Trees();

    Indices *tIndices = new Indices();

    DataTreeVector *tRunDataTrees = new DataTreeVector();
    DataTreeVector *tEventDataTrees = new DataTreeVector();
    DataTreeVector *tTrackDataTrees = new DataTreeVector();
    DataTreeVector *tStepDataTrees = new DataTreeVector();
    DataTreeVector *tOutputRunDataTrees = new DataTreeVector();
    DataTreeVector *tOutputEventDataTrees = new DataTreeVector();
    DataTreeVector *tOutputTrackDataTrees = new DataTreeVector();
    DataTreeVector *tOutputStepDataTrees = new DataTreeVector();

    //maps for data
    vector< DataMap * > tRunDataMapVector;
    vector< DataMap * > tEventDataMapVector;
    vector< DataMap * > tTrackDataMapVector;
    vector< DataMap * > tStepDataMapVector;

    //save number of indices in each data output
    vector< unsigned int > tNumRunDataEntries;
    vector< unsigned int > tNumEventDataEntries;
    vector< unsigned int > tNumTrackDataEntries;
    vector< unsigned int > tNumStepDataEntries;

    KRootFile* tInputRootFile;
    string tInputname;

    //make output file
    KRootFile* tOutputRootFile = new KRootFile();
    string tOutputname( argv[argc-1] );

    //make the output file '.root'
    if( tOutputname.length() < 5 || tOutputname.substr( tOutputname.length()-5, tOutputname.length()-1 ) != ".root")
    {
        tOutputname+=string(".root");
    }
    tOutputRootFile->AddToNames( tOutputname );

    if( tOutputRootFile->Open( KFile::eWrite ) == false )
    {
        mainmsg( eError ) << "Could not make file: <" << tOutputname << "> Exiting..." << eom;
        exit( -1 );
    }

    //set kassiopeia label for output file
    TObjString* fLabel = new TObjString( string( "KASSIOPEIA_TREE_DATA" ).c_str() );
    fLabel->Write( "LABEL", TObject::kOverwrite );
    TTree::SetBranchStyle( 1 );


    //loop over all input files
    for( int i = 0; i < argc - 2; ++i )
    {
        tInputRootFile = new KRootFile();
        tInputname = argv[i+1];
        tInputRootFile->AddToNames( tInputname );
        mainmsg( eNormal ) << "Analyzing file " << i+1 << " of " << argc - 2 << ": <" << tInputname << ">" << eom;

        //read input file
        if( tInputRootFile->Open( KFile::eRead ) == false )
        {
            mainmsg( eError ) << "Could not read file: <" << tInputname << ">. Exiting..." << eom;
            exit( -1 );
        }

        //check input file for kassiopeia label
        CheckLabel( tInputRootFile );

        mainmsg( eDebug ) << "Getting indices and key trees..." << eom;
        //get all index and key trees included in any kassiopeia root file
        tInputTrees->tRunTree = GetTree( tInputRootFile, "RUN_DATA" );
        tInputTrees->tEventTree = GetTree( tInputRootFile, "EVENT_DATA" );
        tInputTrees->tTrackTree = GetTree( tInputRootFile, "TRACK_DATA" );
        tInputTrees->tStepTree = GetTree( tInputRootFile, "STEP_DATA" );
        tInputTrees->tRunKeysTree = GetTree( tInputRootFile, "RUN_KEYS" );
        tInputTrees->tEventKeysTree = GetTree( tInputRootFile, "EVENT_KEYS" );
        tInputTrees->tTrackKeysTree = GetTree( tInputRootFile, "TRACK_KEYS" );
        tInputTrees->tStepKeysTree = GetTree( tInputRootFile, "STEP_KEYS" );

        mainmsg( eDebug ) << "GetIndicesBranchAddresses..." << eom;
        //set all local addresses to the addresses in the input file
        GetIndicesBranchAddresses( tInputTrees, tIndices );

        //create basic output tree structures after reading the first file
        if( i == 0 )
        {
            mainmsg( eDebug ) << "Getting all data trees..." << eom;
            //get all data trees which may be included in the input file
            GetDataTrees( tInputRootFile, tInputTrees->tRunKeysTree, tRunDataTrees, true );
            GetDataTrees( tInputRootFile, tInputTrees->tEventKeysTree, tEventDataTrees, true );
            GetDataTrees( tInputRootFile, tInputTrees->tTrackKeysTree, tTrackDataTrees, true );
            GetDataTrees( tInputRootFile, tInputTrees->tStepKeysTree, tStepDataTrees, true );

            mainmsg( eDebug ) << "Getting all data branches..." << eom;
            //get all data branches in any data trees found before
            for( DataTreeVector::iterator tIt = tRunDataTrees->begin(); tIt != tRunDataTrees->end(); tIt += 3 )
            {
                tRunDataMapVector.push_back( new DataMap() );
                GetDataBranches( (*tIt).second, (*(tIt+1)).second, (*(tIt+2)).second, tRunDataMapVector.back() );
            }
            for( DataTreeVector::iterator tIt = tEventDataTrees->begin(); tIt != tEventDataTrees->end(); tIt += 3 )
            {
                tEventDataMapVector.push_back( new DataMap() );
                GetDataBranches( (*tIt).second, (*(tIt+1)).second, (*(tIt+2)).second, tEventDataMapVector.back() );
            }
            for( DataTreeVector::iterator tIt = tTrackDataTrees->begin(); tIt != tTrackDataTrees->end(); tIt += 3 )
            {
                tTrackDataMapVector.push_back( new DataMap() );
                GetDataBranches( (*tIt).second, (*(tIt+1)).second, (*(tIt+2)).second, tTrackDataMapVector.back() );
            }
            for( DataTreeVector::iterator tIt = tStepDataTrees->begin(); tIt != tStepDataTrees->end(); tIt += 3 )
            {
                tStepDataMapVector.push_back( new DataMap() );
                GetDataBranches( (*tIt).second, (*(tIt+1)).second, (*(tIt+2)).second, tStepDataMapVector.back() );
            }

            //initialize numlength variables
            for( unsigned int tI = 0; tI < tRunDataMapVector.size(); ++tI )
            {
                tNumRunDataEntries.push_back( 0 );
            }
            for( unsigned int tI = 0; tI < tEventDataMapVector.size(); ++tI )
            {
                tNumEventDataEntries.push_back( 0 );
            }
            for( unsigned int tI = 0; tI < tTrackDataMapVector.size(); ++tI )
            {
                tNumTrackDataEntries.push_back( 0 );
            }
            for( unsigned int tI = 0; tI < tStepDataMapVector.size(); ++tI )
            {
                tNumStepDataEntries.push_back( 0 );
            }

            //write the tree structure into the output file
            WriteIndexTrees( tOutputRootFile, tOutputTrees, tIndices, tBufferSize, tSplitSize );

            //get, write and fill the key trees
            GetWriteFillKeyTrees( tOutputRootFile, tInputTrees, tOutputTrees, tBufferSize, tSplitSize );

            //write all data trees which may be included in the input file
            WriteDataTrees( tOutputRootFile, tOutputRunDataTrees, tRunDataTrees );
            WriteDataTrees( tOutputRootFile, tOutputEventDataTrees, tEventDataTrees );
            WriteDataTrees( tOutputRootFile, tOutputTrackDataTrees, tTrackDataTrees );
            WriteDataTrees( tOutputRootFile, tOutputStepDataTrees, tStepDataTrees );

            //write all data branches in any data trees
            for( unsigned int tI = 0; tI < tOutputRunDataTrees->size(); tI += 3 )
            {
                WriteDataBranches( tOutputRunDataTrees->at( tI ).second, tOutputRunDataTrees->at( tI+1 ).second, tOutputRunDataTrees->at( tI+2 ).second, tRunDataMapVector[tI/3], tBufferSize, tSplitSize );
                FillDataStructureBranch( tOutputRunDataTrees->at( tI ).second, tRunDataTrees->at( tI ).second );
            }
            for( unsigned int tI = 0; tI < tOutputEventDataTrees->size(); tI += 3 )
            {
                WriteDataBranches( tOutputEventDataTrees->at( tI ).second, tOutputEventDataTrees->at( tI+1 ).second, tOutputEventDataTrees->at( tI+2 ).second, tEventDataMapVector[tI/3], tBufferSize, tSplitSize );
                FillDataStructureBranch( tOutputEventDataTrees->at( tI ).second, tEventDataTrees->at( tI ).second );
            }
            for( unsigned int tI = 0; tI < tOutputTrackDataTrees->size(); tI += 3 )
            {
                WriteDataBranches( tOutputTrackDataTrees->at( tI ).second, tOutputTrackDataTrees->at( tI+1 ).second, tOutputTrackDataTrees->at( tI+2 ).second, tTrackDataMapVector[tI/3], tBufferSize, tSplitSize );
                FillDataStructureBranch( tOutputTrackDataTrees->at( tI ).second, tTrackDataTrees->at( tI ).second );
            }
            for( unsigned int tI = 0; tI < tOutputStepDataTrees->size(); tI += 3 )
            {
                WriteDataBranches( tOutputStepDataTrees->at( tI ).second, tOutputStepDataTrees->at( tI+1 ).second, tOutputStepDataTrees->at( tI+2 ).second, tStepDataMapVector[tI/3], tBufferSize, tSplitSize );
                FillDataStructureBranch( tOutputStepDataTrees->at( tI ).second, tStepDataTrees->at( tI ).second );
            }
        }
        else
        {
            mainmsg( eDebug ) << "Getting all data trees..." << eom;
            //get all data trees which may be included in the input file
            GetDataTrees( tInputRootFile, tInputTrees->tRunKeysTree, tRunDataTrees, false );
            GetDataTrees( tInputRootFile, tInputTrees->tEventKeysTree, tEventDataTrees, false );
            GetDataTrees( tInputRootFile, tInputTrees->tTrackKeysTree, tTrackDataTrees, false );
            GetDataTrees( tInputRootFile, tInputTrees->tStepKeysTree, tStepDataTrees, false );

            mainmsg( eDebug ) << "Getting all data branches..." << eom;
            //get all data branches in any data trees found before
            unsigned int tIndex = 0;
            for( DataTreeVector::iterator tIt = tRunDataTrees->begin(); tIt != tRunDataTrees->end(); tIt += 3 )
            {
                GetDataBranches( (*tIt).second, (*(tIt+1)).second, (*(tIt+2)).second, tRunDataMapVector.at(tIndex) );
                ++tIndex;
            }
            tIndex = 0;
            for( DataTreeVector::iterator tIt = tEventDataTrees->begin(); tIt != tEventDataTrees->end(); tIt += 3 )
            {
                GetDataBranches( (*tIt).second, (*(tIt+1)).second, (*(tIt+2)).second, tEventDataMapVector.at(tIndex) );
                ++tIndex;
            }
            tIndex = 0;
            for( DataTreeVector::iterator tIt = tTrackDataTrees->begin(); tIt != tTrackDataTrees->end(); tIt += 3 )
            {
                GetDataBranches( (*tIt).second, (*(tIt+1)).second, (*(tIt+2)).second, tTrackDataMapVector.at(tIndex) );
                ++tIndex;
            }
            tIndex = 0;
            for( DataTreeVector::iterator tIt = tStepDataTrees->begin(); tIt != tStepDataTrees->end(); tIt += 3 )
            {
                GetDataBranches( (*tIt).second, (*(tIt+1)).second, (*(tIt+2)).second, tStepDataMapVector.at(tIndex) );
                ++tIndex;
            }
        }

        mainmsg( eDebug ) << "FillIndexTrees..." << eom;
        //put new index data in the combined output file
        FillIndexTrees( tOutputTrees, tInputTrees, tIndices );

        mainmsg( eDebug ) << "FillDataBranches..." << eom;
        //fill data branches in the combined output file
        for( unsigned int tI = 0; tI < tOutputRunDataTrees->size(); tI += 3 )
        {
            FillDataPresenceBranch( tOutputRunDataTrees->at( tI+1 ).second, tRunDataTrees->at( tI+1 ).second, tRunDataMapVector.at( tI/3 ) );
            FillDataBranches( tOutputRunDataTrees->at( tI+2 ).second, tRunDataTrees->at( tI+2 ).second );
        }
        for( unsigned int tI = 0; tI < tOutputEventDataTrees->size(); tI += 3 )
        {
            FillDataPresenceBranch( tOutputEventDataTrees->at( tI+1 ).second, tEventDataTrees->at( tI+1 ).second, tEventDataMapVector.at( tI/3 ) );
            FillDataBranches( tOutputEventDataTrees->at( tI+2 ).second, tEventDataTrees->at( tI+2 ).second );
        }
        for( unsigned int tI = 0; tI < tOutputTrackDataTrees->size(); tI += 3 )
        {
            FillDataPresenceBranch( tOutputTrackDataTrees->at( tI+1 ).second, tTrackDataTrees->at( tI+1 ).second, tTrackDataMapVector.at( tI/3 ) );
            FillDataBranches( tOutputTrackDataTrees->at( tI+2 ).second, tTrackDataTrees->at( tI+2 ).second );
        }
        for( unsigned int tI = 0; tI < tOutputStepDataTrees->size(); tI += 3 )
        {
            FillDataPresenceBranch( tOutputStepDataTrees->at( tI+1 ).second, tStepDataTrees->at( tI+1 ).second, tStepDataMapVector.at( tI/3 ) );
            FillDataBranches( tOutputStepDataTrees->at( tI+2 ).second, tStepDataTrees->at( tI+2 ).second );
        }

        //close file and delete file pointer before analyzing next file
        tInputRootFile->Close();
        delete tInputRootFile;
    }

    mainmsg( eNormal ) << "Writing file: <" << tOutputname << ">. This can take a while..." << eom;

    tOutputRootFile->File()->Write( "", TObject::kOverwrite );
    tOutputRootFile->Close();

    mainmsg( eNormal ) << "Wrote file: <" << tOutputname << ">" << eom;
    mainmsg( eNormal ) << "Merge completed. Exiting..." << eom;
}


TTree* GetTree( KRootFile *tInputRootFile, TString tName )
{
    TTree *tTree = (TTree*) (tInputRootFile->File()->Get( tName ) );
    if( !tTree )
    {
        mainmsg( eError ) << "File <" << tInputRootFile->GetName() << "> has no Tree " << tName << ". Exiting..." << eom;
        exit( -1 );
    }
    return tTree;
}

void CheckLabel( KRootFile *tInputRootFile )
{
    TObjString *tLabel = (TObjString*) (tInputRootFile->File()->Get( "LABEL" ));
    if( !tLabel )
    {
        mainmsg( eError ) << "File <" << tInputRootFile->GetName() << "> has no LABEL. Probably not a Kassiopeia file. Exiting..." << eom;
        exit( -1 );
    }

    if( tLabel->GetString().CompareTo( string( "KASSIOPEIA_TREE_DATA" ).c_str() ) != 0 )
    {
        mainmsg( eError ) << "File's <" << tInputRootFile->GetName() << "> LABEL is not 'KASSIOPEIA_TREE_DATA'. Probably not a Kassiopeia File. Exiting..." << eom;
        exit( -1 );
    }
}

void GetIndicesBranchAddresses( Trees *tTrees, Indices *tIndices )
{
    tTrees->tRunTree->SetBranchAddress( "RUN_INDEX", &(tIndices->tRunIndex) );
    tTrees->tRunTree->SetBranchAddress( "FIRST_EVENT_INDEX", &(tIndices->tRunFirstEvent) );
    tTrees->tRunTree->SetBranchAddress( "LAST_EVENT_INDEX", &(tIndices->tRunLastEvent) );
    tTrees->tRunTree->SetBranchAddress( "FIRST_TRACK_INDEX", &(tIndices->tRunFirstTrack) );
    tTrees->tRunTree->SetBranchAddress( "LAST_TRACK_INDEX", &(tIndices->tRunLastTrack) );
    tTrees->tRunTree->SetBranchAddress( "FIRST_STEP_INDEX", &(tIndices->tRunFirstStep) );
    tTrees->tRunTree->SetBranchAddress( "LAST_STEP_INDEX", &(tIndices->tRunLastStep) );

    tTrees->tEventTree->SetBranchAddress( "EVENT_INDEX", &(tIndices->tEventIndex) );
    tTrees->tEventTree->SetBranchAddress( "FIRST_TRACK_INDEX", &(tIndices->tEventFirstTrack) );
    tTrees->tEventTree->SetBranchAddress( "LAST_TRACK_INDEX", &(tIndices->tEventLastTrack) );
    tTrees->tEventTree->SetBranchAddress( "FIRST_STEP_INDEX", &(tIndices->tEventFirstStep) );
    tTrees->tEventTree->SetBranchAddress( "LAST_STEP_INDEX", &(tIndices->tEventLastStep) );

    tTrees->tTrackTree->SetBranchAddress( "TRACK_INDEX", &(tIndices->tTrackIndex) );
    tTrees->tTrackTree->SetBranchAddress( "FIRST_STEP_INDEX", &(tIndices->tTrackFirstStep ) );
    tTrees->tTrackTree->SetBranchAddress( "LAST_STEP_INDEX", &(tIndices->tTrackLastStep) );

    tTrees->tStepTree->SetBranchAddress( "STEP_INDEX", &(tIndices->tStepIndex) );
}

void GetDataTrees( KRootFile *tInputFile, TTree *tKeyTree, DataTreeVector *tDataTreeVector, bool first )
{
    string tKey;
    string* tKeyPointer = &tKey;
    string** tKeyHandle = &tKeyPointer;
    string tDataTreeName;

    tKeyTree->SetBranchAddress( "KEY", tKeyHandle );
    for( Long64_t tKeyIndex = 0; tKeyIndex < tKeyTree->GetEntries(); tKeyIndex++ )
    {
        tKeyTree->GetEntry( tKeyIndex );

        tDataTreeName = tKey + string( "_STRUCTURE" );
        if( first )
            tDataTreeVector->push_back( make_pair( tDataTreeName, ( TTree * ) tInputFile->File()->Get( tDataTreeName.c_str() ) ) );
        else
            tDataTreeVector->at(3*tKeyIndex) = ( make_pair( tDataTreeName, ( TTree * ) tInputFile->File()->Get( tDataTreeName.c_str() ) ) );

        tDataTreeName = tKey + string( "_PRESENCE" );
        if( first )
            tDataTreeVector->push_back( make_pair( tDataTreeName, ( TTree * ) tInputFile->File()->Get( tDataTreeName.c_str() ) ) );
        else
            tDataTreeVector->at((3*tKeyIndex)+1) = ( make_pair( tDataTreeName, ( TTree * ) tInputFile->File()->Get( tDataTreeName.c_str() ) ) );

        tDataTreeName = tKey + string( "_DATA" );
        if( first )
            tDataTreeVector->push_back( make_pair( tDataTreeName, ( TTree * ) tInputFile->File()->Get( tDataTreeName.c_str() ) ) );
        else
            tDataTreeVector->at((3*tKeyIndex)+2) = ( make_pair( tDataTreeName, ( TTree * ) tInputFile->File()->Get( tDataTreeName.c_str() ) ) );
    }
}

void GetDataBranches( TTree *tStructureTree, TTree *tPresenceTree, TTree *tDataTree, DataMap *tDataMap )
{
    mainmsg( eDebug ) << "StructureTree: " << tStructureTree->GetName() << eom;
    mainmsg( eDebug ) << "PresenceTree: " << tPresenceTree->GetName() << eom;
    mainmsg( eDebug ) << "DataTree: " << tDataTree->GetName() << eom;
    tStructureTree->SetBranchAddress( "LABEL", tDataMap->tStructureLabelHandle );
    tStructureTree->SetBranchAddress( "TYPE", tDataMap->tStructureTypeHandle );

    for( Long64_t tStructureIndex = 0; tStructureIndex < tStructureTree->GetEntries(); tStructureIndex++ )
    {
        tStructureTree->GetEntry( tStructureIndex );
        mainmsg( eDebug ) << "analyzing structure with label <" << tDataMap->tStructureLabel << "> and type <" << tDataMap->tStructureType << ">" << eom;

        if( tDataMap->tStructureType == string( "bool" ) )
        {
            Bool_t &tBool = tDataMap->tBoolMap[ tDataMap->tStructureLabel ];
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tBool );
            continue;
        }
        if( tDataMap->tStructureType == string( "unsigned_char" ) )
        {
            UChar_t &tUChar = tDataMap->tUCharMap[ tDataMap->tStructureLabel ];
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tUChar );
            continue;
        }
        if( tDataMap->tStructureType == string( "char" ) )
        {
            Char_t &tChar = tDataMap->tCharMap[ tDataMap->tStructureLabel ];
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tChar );
            continue;
        }
        if( tDataMap->tStructureType == string( "unsigned_short" ) )
        {
            UShort_t &tUShort = tDataMap->tUShortMap[ tDataMap->tStructureLabel ];
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tUShort );
            continue;
        }
        if( tDataMap->tStructureType == string( "short" ) )
        {
            Short_t &tShort = tDataMap->tShortMap[ tDataMap->tStructureLabel ];
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tShort );
            continue;
        }
        if( tDataMap->tStructureType == string( "unsigned_int" ) )
        {
            UInt_t &tUInt = tDataMap->tUIntMap[ tDataMap->tStructureLabel ];
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tUInt );
            continue;
        }
        if( tDataMap->tStructureType == string( "int" ) )
        {
            Int_t &tInt = tDataMap->tIntMap[ tDataMap->tStructureLabel ];
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tInt );
            continue;
        }
        if( tDataMap->tStructureType == string( "unsigned_long" ) )
        {
            ULong64_t &tULong = tDataMap->tULongMap[ tDataMap->tStructureLabel ];
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tULong );
            continue;
        }
        if( tDataMap->tStructureType == string( "long" ) )
        {
            Long64_t &tLong = tDataMap->tLongMap[ tDataMap->tStructureLabel ];
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tLong );
            continue;
        }
        if( tDataMap->tStructureType == string( "float" ) )
        {
            Float_t &tFloat = tDataMap->tFloatMap[ tDataMap->tStructureLabel ];
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tFloat );
            continue;
        }
        if( tDataMap->tStructureType == string( "double" ) )
        {
            Double_t &tDouble = tDataMap->tDoubleMap[ tDataMap->tStructureLabel ];
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tDouble );
            continue;
        }
        if( tDataMap->tStructureType == string( "string" ) )
        {
            string &tString = tDataMap->tStringMap[ tDataMap->tStructureLabel ];
            string* &tStringPointer = tDataMap->tStringPointerMap[ tDataMap->tStructureLabel ];
            tStringPointer = &tString;
            tDataTree->SetBranchAddress( tDataMap->tStructureLabel.c_str(), &tStringPointer );
            continue;
        }
        if( tDataMap->tStructureType == string( "two_vector" ) )
        {
            Double_t &tDoubleX = tDataMap->tTwoVectorXMap[ tDataMap->tStructureLabel + string( "_x" ) ];
            Double_t &tDoubleY = tDataMap->tTwoVectorYMap[ tDataMap->tStructureLabel + string( "_y" ) ];
            tDataTree->SetBranchAddress( ( tDataMap->tStructureLabel + string( "_x" ) ).c_str(), &tDoubleX );
            tDataTree->SetBranchAddress( ( tDataMap->tStructureLabel + string( "_y" ) ).c_str(), &tDoubleY );
            continue;
        }
        if( tDataMap->tStructureType == string( "three_vector" ) )
        {
            Double_t &tDoubleX = tDataMap->tThreeVectorXMap[ tDataMap->tStructureLabel + string( "_x" ) ];
            Double_t &tDoubleY = tDataMap->tThreeVectorYMap[ tDataMap->tStructureLabel + string( "_y" ) ];
            Double_t &tDoubleZ = tDataMap->tThreeVectorZMap[ tDataMap->tStructureLabel + string( "_z" ) ];
            tDataTree->SetBranchAddress( ( tDataMap->tStructureLabel + string( "_x" ) ).c_str(), &tDoubleX );
            tDataTree->SetBranchAddress( ( tDataMap->tStructureLabel + string( "_y" ) ).c_str(), &tDoubleY );
            tDataTree->SetBranchAddress( ( tDataMap->tStructureLabel + string( "_z" ) ).c_str(), &tDoubleZ );
            continue;
        }

        mainmsg( eError ) << "could not analyze branch with label <" << tDataMap->tStructureLabel << "> and type <" << tDataMap->tStructureType << ">. Exiting..." << eom;
        exit( -1 );
    }

    tPresenceTree->SetBranchAddress( "INDEX", tDataMap->tPresenceIndexPointer );
    tPresenceTree->SetBranchAddress( "LENGTH", tDataMap->tPresenceLengthPointer );
    tPresenceTree->GetEntry( 0 );
}

void WriteIndexTrees( KRootFile *tRootFile, Trees *tTrees, Indices *tIndices, const int tBufferSize, const int tSplitSize )
{
    tTrees->tRunTree = new TTree( "RUN_DATA", "RUN_DATA" );
    tTrees->tRunTree->SetDirectory( tRootFile->File() );
    tTrees->tRunTree->Branch( "RUN_INDEX", &(tIndices->tRunIndex), tBufferSize, tSplitSize );
    tTrees->tRunTree->Branch( "FIRST_STEP_INDEX", &(tIndices->tRunFirstStep), tBufferSize, tSplitSize );
    tTrees->tRunTree->Branch( "LAST_STEP_INDEX", &(tIndices->tRunLastStep), tBufferSize, tSplitSize );
    tTrees->tRunTree->Branch( "FIRST_TRACK_INDEX", &(tIndices->tRunFirstTrack), tBufferSize, tSplitSize );
    tTrees->tRunTree->Branch( "LAST_TRACK_INDEX", &(tIndices->tRunLastTrack), tBufferSize, tSplitSize );
    tTrees->tRunTree->Branch( "FIRST_EVENT_INDEX", &(tIndices->tRunFirstEvent), tBufferSize, tSplitSize );
    tTrees->tRunTree->Branch( "LAST_EVENT_INDEX", &(tIndices->tRunLastEvent), tBufferSize, tSplitSize );

    tTrees->tEventTree = new TTree( "EVENT_DATA", "EVENT_DATA" );
    tTrees->tEventTree->SetDirectory( tRootFile->File() );
    tTrees->tEventTree->Branch( "EVENT_INDEX", &(tIndices->tEventIndex), tBufferSize, tSplitSize );
    tTrees->tEventTree->Branch( "FIRST_STEP_INDEX", &(tIndices->tEventFirstStep), tBufferSize, tSplitSize );
    tTrees->tEventTree->Branch( "LAST_STEP_INDEX", &(tIndices->tEventLastStep), tBufferSize, tSplitSize );
    tTrees->tEventTree->Branch( "FIRST_TRACK_INDEX", &(tIndices->tEventFirstTrack), tBufferSize, tSplitSize );
    tTrees->tEventTree->Branch( "LAST_TRACK_INDEX", &(tIndices->tEventLastTrack), tBufferSize, tSplitSize );

    tTrees->tTrackTree = new TTree( "TRACK_DATA", "TRACK_DATA" );
    tTrees->tTrackTree->SetDirectory( tRootFile->File() );
    tTrees->tTrackTree->Branch( "TRACK_INDEX", &(tIndices->tTrackIndex), tBufferSize, tSplitSize );
    tTrees->tTrackTree->Branch( "FIRST_STEP_INDEX", &(tIndices->tTrackFirstStep), tBufferSize, tSplitSize );
    tTrees->tTrackTree->Branch( "LAST_STEP_INDEX", &(tIndices->tTrackLastStep), tBufferSize, tSplitSize );

    tTrees->tStepTree = new TTree( "STEP_DATA", "STEP_DATA" );
    tTrees->tStepTree->SetDirectory( tRootFile->File() );
    tTrees->tStepTree->Branch( "STEP_INDEX", &(tIndices->tStepIndex), tBufferSize, tSplitSize );
}

void WriteDataTrees( KRootFile *tRootFile, DataTreeVector *tOutputDataTreeVector, DataTreeVector *tDataTreeVector )
{
    for( DataTreeVector::iterator tIt = tDataTreeVector->begin(); tIt != tDataTreeVector->end(); ++tIt )
    {
        tOutputDataTreeVector->push_back( make_pair( (*tIt).first, new TTree( (TString) (*tIt).first, (TString) (*tIt).first ) ) );
        tOutputDataTreeVector->back().second->SetDirectory( tRootFile->File() );
    }
}

void WriteDataBranches( TTree *tStructureTree, TTree *tPresenceTree, TTree *tDataTree, DataMap *tDataMap, const int tBufferSize, const int tSplitSize )
{
    tStructureTree->Branch( "LABEL", tDataMap->tStructureLabelPointer, tBufferSize, tSplitSize );
    tStructureTree->Branch( "TYPE", tDataMap->tStructureTypePointer, tBufferSize, tSplitSize );

    tPresenceTree->Branch( "INDEX", tDataMap->tPresenceIndexPointer, tBufferSize, tSplitSize );
    tPresenceTree->Branch( "LENGTH", tDataMap->tPresenceLengthPointer, tBufferSize, tSplitSize );

    for( map< string, Bool_t >::iterator tIt = tDataMap->tBoolMap.begin(); tIt != tDataMap->tBoolMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, UChar_t >::iterator tIt = tDataMap->tUCharMap.begin(); tIt != tDataMap->tUCharMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, Char_t >::iterator tIt = tDataMap->tCharMap.begin(); tIt != tDataMap->tCharMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, UShort_t >::iterator tIt = tDataMap->tUShortMap.begin(); tIt != tDataMap->tUShortMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, Short_t >::iterator tIt = tDataMap->tShortMap.begin(); tIt != tDataMap->tShortMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, UInt_t >::iterator tIt = tDataMap->tUIntMap.begin(); tIt != tDataMap->tUIntMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, Int_t >::iterator tIt = tDataMap->tIntMap.begin(); tIt != tDataMap->tIntMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, ULong64_t >::iterator tIt = tDataMap->tULongMap.begin(); tIt != tDataMap->tULongMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, Long64_t >::iterator tIt = tDataMap->tLongMap.begin(); tIt != tDataMap->tLongMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, Float_t >::iterator tIt = tDataMap->tFloatMap.begin(); tIt != tDataMap->tFloatMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, Double_t >::iterator tIt = tDataMap->tDoubleMap.begin(); tIt != tDataMap->tDoubleMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, string * >::iterator tIt = tDataMap->tStringPointerMap.begin(); tIt != tDataMap->tStringPointerMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, Double_t >::iterator tIt = tDataMap->tTwoVectorXMap.begin(); tIt != tDataMap->tTwoVectorXMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, Double_t >::iterator tIt = tDataMap->tTwoVectorYMap.begin(); tIt != tDataMap->tTwoVectorYMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, Double_t >::iterator tIt = tDataMap->tThreeVectorXMap.begin(); tIt != tDataMap->tThreeVectorXMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, Double_t >::iterator tIt = tDataMap->tThreeVectorYMap.begin(); tIt != tDataMap->tThreeVectorYMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
    for( map< string, Double_t >::iterator tIt = tDataMap->tThreeVectorZMap.begin(); tIt != tDataMap->tThreeVectorZMap.end(); ++tIt )
    {
        tDataTree->Branch( (TString) (*tIt).first, &((*tIt).second), tBufferSize, tSplitSize );
    }
}

void FillDataStructureBranch( TTree *tOutputStructureTree, TTree *tInputStructureTree )
{
    for( Int_t tIndex = 0; tIndex < tInputStructureTree->GetEntries(); tIndex++ )
    {
        tInputStructureTree->GetEntry( tIndex );
        tOutputStructureTree->Fill();
    }
}

void GetWriteFillKeyTrees( KRootFile *tRootFile, Trees *tInputTrees, Trees *tOutputTrees, const int tBufferSize, const int tSplitSize )
{
    string tRunKey;
    string* tRunKeyPointer = &tRunKey;
    string** tRunKeyHandle = &tRunKeyPointer;

    string tEventKey;
    string* tEventKeyPointer = &tEventKey;
    string** tEventKeyHandle = &tEventKeyPointer;

    string tTrackKey;
    string* tTrackKeyPointer = &tTrackKey;
    string** tTrackKeyHandle = &tTrackKeyPointer;

    string tStepKey;
    string* tStepKeyPointer = &tStepKey;
    string** tStepKeyHandle = &tStepKeyPointer;

    tInputTrees->tRunKeysTree->SetBranchAddress( "KEY", tRunKeyHandle );
    tInputTrees->tEventKeysTree->SetBranchAddress( "KEY", tEventKeyHandle );
    tInputTrees->tTrackKeysTree->SetBranchAddress( "KEY", tTrackKeyHandle );
    tInputTrees->tStepKeysTree->SetBranchAddress( "KEY", tStepKeyHandle );

    tOutputTrees->tRunKeysTree = new TTree( "RUN_KEYS", "RUN_KEYS" );
    tOutputTrees->tRunKeysTree->SetDirectory( tRootFile->File() );
    tOutputTrees->tRunKeysTree->Branch( "KEY", tRunKeyPointer, tBufferSize, tSplitSize );

    tOutputTrees->tEventKeysTree = new TTree( "EVENT_KEYS", "EVENT_KEYS" );
    tOutputTrees->tEventKeysTree->SetDirectory( tRootFile->File() );
    tOutputTrees->tEventKeysTree->Branch( "KEY", tEventKeyPointer, tBufferSize, tSplitSize );

    tOutputTrees->tTrackKeysTree = new TTree( "TRACK_KEYS", "TRACK_KEYS" );
    tOutputTrees->tTrackKeysTree->SetDirectory( tRootFile->File() );
    tOutputTrees->tTrackKeysTree->Branch( "KEY", tTrackKeyPointer, tBufferSize, tSplitSize );

    tOutputTrees->tStepKeysTree = new TTree( "STEP_KEYS", "STEP_KEYS" );
    tOutputTrees->tStepKeysTree->SetDirectory( tRootFile->File() );
    tOutputTrees->tStepKeysTree->Branch( "KEY", tStepKeyPointer, tBufferSize, tSplitSize );

    for( Int_t tIndex = 0; tIndex < tInputTrees->tRunKeysTree->GetEntries(); tIndex++ )
    {
        tInputTrees->tRunKeysTree->GetEntry( tIndex );
        tOutputTrees->tRunKeysTree->Fill();
    }
    for( Int_t tIndex = 0; tIndex < tInputTrees->tEventKeysTree->GetEntries(); tIndex++ )
    {
        tInputTrees->tEventKeysTree->GetEntry( tIndex );
        tOutputTrees->tEventKeysTree->Fill();
    }
    for( Int_t tIndex = 0; tIndex < tInputTrees->tTrackKeysTree->GetEntries(); tIndex++ )
    {
        tInputTrees->tTrackKeysTree->GetEntry( tIndex );
        tOutputTrees->tTrackKeysTree->Fill();
    }
    for( Int_t tIndex = 0; tIndex < tInputTrees->tStepKeysTree->GetEntries(); tIndex++ )
    {
        tInputTrees->tStepKeysTree->GetEntry( tIndex );
        tOutputTrees->tStepKeysTree->Fill();
    }
}

void FillIndexTrees( Trees *tOutputTrees, Trees *tInputTrees, Indices *tIndices )
{
    unsigned int tRunEntries = tOutputTrees->tRunTree->GetEntries();
    unsigned int tEventEntries = tOutputTrees->tEventTree->GetEntries();
    unsigned int tTrackEntries = tOutputTrees->tTrackTree->GetEntries();
    unsigned int tStepEntries = tOutputTrees->tStepTree->GetEntries();

    for( Int_t tIndex = 0; tIndex < tInputTrees->tRunTree->GetEntries(); tIndex++ )
    {
        tInputTrees->tRunTree->GetEntry( tIndex );

        (tIndices->tRunIndex) += tRunEntries;
        (tIndices->tRunFirstEvent) += tEventEntries;
        (tIndices->tRunLastEvent) += tEventEntries;
        (tIndices->tRunFirstTrack) += tTrackEntries;
        (tIndices->tRunLastTrack) += tTrackEntries;
        (tIndices->tRunFirstStep) += tStepEntries;
        (tIndices->tRunLastStep) += tStepEntries;

        tOutputTrees->tRunTree->Fill();
    }
    for( Int_t tIndex = 0; tIndex < tInputTrees->tEventTree->GetEntries(); tIndex++ )
    {
        tInputTrees->tEventTree->GetEntry( tIndex );

        (tIndices->tEventIndex) += tEventEntries;
        (tIndices->tEventFirstTrack) += tTrackEntries;
        (tIndices->tEventLastTrack) += tTrackEntries;
        (tIndices->tEventFirstStep) += tStepEntries;
        (tIndices->tEventLastStep) += tStepEntries;

        tOutputTrees->tEventTree->Fill();
    }
    for( Int_t tIndex = 0; tIndex < tInputTrees->tTrackTree->GetEntries(); tIndex++ )
    {
        tInputTrees->tTrackTree->GetEntry( tIndex );

        (tIndices->tTrackIndex) += tTrackEntries;
        (tIndices->tTrackFirstStep) += tStepEntries;
        (tIndices->tTrackLastStep) += tStepEntries;

        tOutputTrees->tTrackTree->Fill();
    }
    for( Int_t tIndex = 0; tIndex < tInputTrees->tStepTree->GetEntries(); tIndex++ )
    {
        tInputTrees->tStepTree->GetEntry( tIndex );

        (tIndices->tStepIndex) += tStepEntries;

        tOutputTrees->tStepTree->Fill();
    }
}

void FillDataBranches( TTree *tOutputDataTree, TTree *tInputDataTree )
{
    for( Int_t tIndex = 0; tIndex < tInputDataTree->GetEntries(); tIndex++ )
    {
        tInputDataTree->GetEntry( tIndex );
        tOutputDataTree->Fill();
    }
}

void FillDataPresenceBranch( TTree *tOutputDataPresenceTree, TTree *tInputDataPresenceTree, DataMap *tDataMap )
{
	mainmsg( eDebug ) << "Filling presence tree"<<eom;

	unsigned int tLastEntry = 0;
    unsigned int tLastLength = 0;

	//get the last presence index and length in the output tree, if this is not the first presence data
	if ( tOutputDataPresenceTree->GetEntries() != 0)
	{
		tOutputDataPresenceTree->GetEntry( tOutputDataPresenceTree->GetEntries() - 1 );
	    tLastEntry = (tDataMap->tPresenceIndex);
	    tLastLength = (tDataMap->tPresenceLength);
	}
    mainmsg( eDebug ) <<"last entry is: "<<tLastEntry<<eom;
    mainmsg( eDebug ) <<"last length is: "<<tLastLength<<eom;

    for( Int_t tIndex = 0; tIndex < tInputDataPresenceTree->GetEntries(); tIndex++ )
    {
        tInputDataPresenceTree->GetEntry( tIndex );
        (tDataMap->tPresenceIndex) += ( tLastEntry + tLastLength);
        tOutputDataPresenceTree->Fill();
    }
}

