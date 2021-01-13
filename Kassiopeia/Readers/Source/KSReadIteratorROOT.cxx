#include "KSReadIteratorROOT.h"

using namespace std;

namespace Kassiopeia
{

KSReadIteratorROOT::KSReadIteratorROOT(TFile* aFile, TTree* aKeyTree, TTree* aDataTree) :
    fData(aDataTree),
    fValid(false),
    fIndex(0),
    fFirstIndex(0),
    fLastIndex(aDataTree->GetEntries() - 1)
{
    string tKey;
    string* tKeyPointer = &tKey;
    string** tKeyHandle = &tKeyPointer;
    Long64_t tKeyIndex;

    TTree* tData;
    string tDataName;

    TTree* tStructure;
    string tStructureName;

    TTree* tPresence;
    string tPresenceName;

    KSReadObjectROOT* tObject;

    aKeyTree->SetBranchAddress("KEY", tKeyHandle);
    for (tKeyIndex = 0; tKeyIndex < aKeyTree->GetEntries(); tKeyIndex++) {
        aKeyTree->GetEntry(tKeyIndex);

        tStructureName = tKey + string("_STRUCTURE");
        tStructure = (TTree*) (aFile->Get(tStructureName.c_str()));

        tPresenceName = tKey + string("_PRESENCE");
        tPresence = (TTree*) (aFile->Get(tPresenceName.c_str()));

        tDataName = tKey + string("_DATA");
        tData = (TTree*) (aFile->Get(tDataName.c_str()));

        tObject = new KSReadObjectROOT(tStructure, tPresence, tData);
        fObjects.insert(ObjectEntry(tKey, tObject));
    }
}
KSReadIteratorROOT::~KSReadIteratorROOT() = default;

void KSReadIteratorROOT::operator<<(const unsigned int& aValue)
{
    fIndex = aValue;

    if (fIndex < fFirstIndex) {
        fValid = false;
        return;
    }

    if (fIndex > fLastIndex) {
        fValid = false;
        return;
    }

    fData->GetEntry(fIndex);
    fValid = true;

    for (auto& object : fObjects) {
        (*(object.second)) << aValue;
    }

    return;
}
void KSReadIteratorROOT::operator++(int)
{
    fIndex++;

    if (fIndex > fLastIndex) {
        fValid = false;
        return;
    }

    fData->GetEntry(fIndex);
    fValid = true;

    for (auto& object : fObjects) {
        (*(object.second))++;
    }

    return;
}
void KSReadIteratorROOT::operator--(int)
{
    fIndex--;
    if (fIndex < fFirstIndex) {
        fValid = false;
        return;
    }

    fData->GetEntry(fIndex);
    fValid = true;

    for (auto& object : fObjects) {
        (*(object.second))--;
    }

    return;
}

bool KSReadIteratorROOT::Valid() const
{
    return fValid;
}
unsigned int KSReadIteratorROOT::Index() const
{
    return fIndex;
}
bool KSReadIteratorROOT::operator<(const unsigned int& aValue) const
{
    return (fIndex < aValue);
}
bool KSReadIteratorROOT::operator<=(const unsigned int& aValue) const
{
    return (fIndex <= aValue);
}
bool KSReadIteratorROOT::operator>(const unsigned int& aValue) const
{
    return (fIndex > aValue);
}
bool KSReadIteratorROOT::operator>=(const unsigned int& aValue) const
{
    return (fIndex >= aValue);
}
bool KSReadIteratorROOT::operator==(const unsigned int& aValue) const
{
    return (fIndex == aValue);
}
bool KSReadIteratorROOT::operator!=(const unsigned int& aValue) const
{
    return (fIndex != aValue);
}

bool KSReadIteratorROOT::HasObject(const string& aLabel)
{
    auto tIt = fObjects.find(aLabel);
    if (tIt != fObjects.end()) {
        return true;
    }
    return false;
}

KSReadObjectROOT& KSReadIteratorROOT::GetObject(const string& aLabel)
{
    auto tIt = fObjects.find(aLabel);
    if (tIt != fObjects.end()) {
        return (*tIt->second);
    }
    readermsg(eError) << "no object named <" << aLabel << ">" << eom;
    return (*tIt->second);
}

const KSReadObjectROOT& KSReadIteratorROOT::GetObject(const string& aLabel) const
{
    auto tIt = fObjects.find(aLabel);
    if (tIt != fObjects.end()) {
        return (*tIt->second);
    }
    readermsg(eError) << "no object named <" << aLabel << ">" << eom;
    return (*tIt->second);
}

}  // namespace Kassiopeia
