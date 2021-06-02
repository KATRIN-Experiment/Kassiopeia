#include "KSWriteASCII.h"

#include "KSComponentGroup.h"
#include "KSWritersMessage.h"

#ifdef Kassiopeia_USE_BOOST
#include "KPathUtils.h"
#endif

using namespace std;
using namespace katrin;
using KGeoBag::KTwoVector;
using KGeoBag::KThreeVector;
using KGeoBag::KTwoMatrix;
using KGeoBag::KThreeMatrix;

namespace Kassiopeia
{

const int KSWriteASCII::fBufferSize = 64000;
const int KSWriteASCII::fSplitLevel = 99;
const string KSWriteASCII::fLabel = string("KASSIOPEIA_TREE_DATA");

KSWriteASCII::Data::Objekt::Objekt(KSComponent* aComponent, string aType, int aPrecision)
{
    fComponent = aComponent;
    fType = aType;
    fPrecision = aPrecision;
}

string KSWriteASCII::Data::Objekt::getValue()
{
    stringstream s;
    s << std::setprecision(fPrecision);
    if (fType == (string) "string")
        s << *(fComponent->As<string>()) << "\t";
    else if (fType == (string) "bool")
        s << *(fComponent->As<bool>()) << "\t";
    else if (fType == (string) "unsigned char")
        s << *(fComponent->As<unsigned char>()) << "\t";
    else if (fType == (string) "char")
        s << *(fComponent->As<char>()) << "\t";
    else if (fType == (string) "unsigned short")
        s << *(fComponent->As<unsigned short>()) << "\t";
    else if (fType == (string) "short")
        s << *(fComponent->As<short>()) << "\t";
    else if (fType == (string) "unsigned int")
        s << *(fComponent->As<unsigned int>()) << "\t";
    else if (fType == (string) "int")
        s << *(fComponent->As<int>()) << "\t";
    else if (fType == (string) "unsigned long")
        s << *(fComponent->As<unsigned long>()) << "\t";
    else if (fType == (string) "long")
        s << *(fComponent->As<long>()) << "\t";
    else if (fType == (string) "long long")
        s << *(fComponent->As<long long>()) << "\t";
    else if (fType == (string) "float")
        s << *(fComponent->As<float>()) << "\t";
    else if (fType == (string) "double")
        s << *(fComponent->As<double>()) << "\t";
    else if (fType == (string) "KThreeVector")
        s << fComponent->As<KThreeVector>()->X() << "\t"
          << fComponent->As<KThreeVector>()->Y() << "\t"
          << fComponent->As<KThreeVector>()->Z() << "\t";
    else if (fType == (string) "KTwoVector")
        s << fComponent->As<KTwoVector>()->X() << "\t"
          << fComponent->As<KTwoVector>()->Y() << "\t";
    else if (fType == (string) "KTwoMatrix")
        for (unsigned i = 0; i < 4; i++)
            s << *(fComponent->As<KTwoMatrix>())[i] << "\t";
    else if (fType == (string) "KThreeMatrix")
        for (unsigned i = 0; i < 9; i++)
            s << *(fComponent->As<KThreeMatrix>())[i] << "\t";
    else
        s << "\t"
          << "\t";

    return s.str();
}

KSWriteASCII::Data::Data(KSComponent* aComponent) :
    fLabel(""),
    fType(""),
    fWriter(),
    fIndex(0),
    fLength(0),
    fComponents()
{
    MakeTitle(aComponent, 0);
}

KSWriteASCII::Data::Data(KSComponent* aComponent, KSWriteASCII* aWriter) :
    fLabel(""),
    fType(""),
    fIndex(0),
    fLength(0),
    fComponents()
{
    fWriter = aWriter;
    MakeTitle(aComponent, 0);
}

KSWriteASCII::Data::~Data() = default;

void KSWriteASCII::Data::Start(const unsigned int& anIndex)
{
    fIndex = anIndex;
    fLength = 0;
    return;
}

void KSWriteASCII::Data::Fill()
{
    KSComponent* tComponent;
    Objekt* tObjekt;
    vector<KSComponent*>::iterator tIt;

    for (tIt = fComponents.begin(); tIt != fComponents.end(); ++tIt) {
        tComponent = (*tIt);
        tComponent->PullUpdate();
    }

    string str;
    for (auto& objekt : fObjekts) {
        tObjekt = objekt;
        str = tObjekt->getValue();

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);
    }

    for (tIt = fComponents.begin(); tIt != fComponents.end(); ++tIt) {
        tComponent = (*tIt);
        tComponent->PullDeupdate();
    }

    fLength++;
    return;
}

void KSWriteASCII::Data::MakeTitle(KSComponent* aComponent, int aTrack)
{
    wtrmsg_debug("making title for object <" << aComponent->GetName() << ">" << eom);

    auto* tComponentGroup = aComponent->As<KSComponentGroup>();
    if (tComponentGroup != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a group" << eom);
        for (unsigned int tIndex = 0; tIndex < tComponentGroup->ComponentCount(); tIndex++)
            MakeTitle(tComponentGroup->ComponentAt(tIndex), aTrack);

        return;
    }

    auto* tString = aComponent->As<string>();
    if (tString != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a string" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "string", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }

        return;
    }

    auto* tTwoVector = aComponent->As<KTwoVector>();
    if (tTwoVector != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a two_vector" << eom);
        string str = (aComponent->GetName() + string("_x") + '\t' + aComponent->GetName() + string("_y") + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "KTwoVector", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }
    auto* tThreeVector = aComponent->As<KThreeVector>();
    if (tThreeVector != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a three_vector" << eom);
        string str = (aComponent->GetName() + string("_x") + '\t' + aComponent->GetName() + string("_y") + '\t' +
                      aComponent->GetName() + string("_z") + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "KThreeVector", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tTwoMatrix = aComponent->As<KTwoMatrix>();
    if (tTwoMatrix != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a two_matrix" << eom);
        string str = (aComponent->GetName() + string("_xx") + '\t' + aComponent->GetName() + string("_xy") +
                      aComponent->GetName() + string("_yx") + '\t' + aComponent->GetName() + string("_yy"));

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "KTwoMatrix", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }
    auto* tThreeMatrix = aComponent->As<KThreeMatrix>();
    if (tThreeMatrix != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a three_matrix" << eom);
        string str = (aComponent->GetName() + string("_xx") + '\t' + aComponent->GetName() + string("_xy") + '\t' + aComponent->GetName() + string("_xz") + '\t' +
                      aComponent->GetName() + string("_yx") + '\t' + aComponent->GetName() + string("_yy") + '\t' + aComponent->GetName() + string("_yz") + '\t' +
                      aComponent->GetName() + string("_zx") + '\t' + aComponent->GetName() + string("_zy") + '\t' + aComponent->GetName() + string("_zz") + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "KThreeMatrix", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tBool = aComponent->As<bool>();
    if (tBool != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a bool" << eom);
       string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "bool", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tUChar = aComponent->As<unsigned char>();
    if (tUChar != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is an unsigned_char" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "unsigned char", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tChar = aComponent->As<char>();
    if (tChar != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a char" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "char", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tUShort = aComponent->As<unsigned short>();
    if (tUShort != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is an unsigned_short" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "unsigned short", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tShort = aComponent->As<short>();
    if (tShort != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a short" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "short", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tUInt = aComponent->As<unsigned int>();
    if (tUInt != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a unsigned_int" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "unsigned int", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tInt = aComponent->As<int>();
    if (tInt != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is an int" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "int", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tULong = aComponent->As<unsigned long>();
    if (tULong != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is an unsigned_long" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "unsigned long", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tLong = aComponent->As<long>();
    if (tLong != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a long" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "long", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tLongLong = aComponent->As<long long>();
    if (tLongLong != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a long_long" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "long_long", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tFloat = aComponent->As<float>();
    if (tFloat != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a float" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "float", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    auto* tDouble = aComponent->As<double>();
    if (tDouble != nullptr) {
        wtrmsg_debug("  object <" << aComponent->GetName() << "> is a double" << eom);
        string str = (aComponent->GetName() + '\t');

        for (char& it : str)
            fWriter->TextFile()->File()->put(it);

        if (aTrack == 0) {
            auto* obj = new Objekt(aComponent, "double", fWriter->Precision());
            fObjekts.push_back(obj);
            fComponents.push_back(aComponent);
        }
        return;
    }

    wtrmsg(eError) << "ASCII writer cannot add object <" << aComponent->GetName() << ">" << eom;

    return;
}

KSWriteASCII::KSWriteASCII() :
    fBase(""),
    fPath(""),
    fStepIteration(1),
    fStepIterationIndex(0),
    fTextFile(nullptr),
    fRunComponents(),
    fActiveRunComponents(),
    fRunIndex(0),
    fRunFirstEventIndex(0),
    fRunLastEventIndex(0),
    fRunFirstTrackIndex(0),
    fRunLastTrackIndex(0),
    fRunFirstStepIndex(0),
    fRunLastStepIndex(0),
    fEventComponents(),
    fActiveEventComponents(),
    fEventIndex(0),
    fEventFirstTrackIndex(0),
    fEventLastTrackIndex(0),
    fEventFirstStepIndex(0),
    fEventLastStepIndex(0),
    fTrackComponents(),
    fActiveTrackComponents(),
    fTrackIndex(0),
    fTrackFirstStepIndex(0),
    fTrackLastStepIndex(0),
    fStepComponent(false),
    fStepComponents(),
    fActiveStepComponents(),
    fStepIndex(0)
{
    fPrecision = std::cout.precision();
}

KSWriteASCII::KSWriteASCII(const KSWriteASCII& aCopy) :
    KSComponent(aCopy),
    fBase(aCopy.fBase),
    fPath(aCopy.fPath),
    fStepIteration(aCopy.fStepIteration),
    fStepIterationIndex(0),
    fTextFile(nullptr),
    fRunComponents(),
    fActiveRunComponents(),
    fRunIndex(0),
    fRunFirstEventIndex(0),
    fRunLastEventIndex(0),
    fRunFirstTrackIndex(0),
    fRunLastTrackIndex(0),
    fRunFirstStepIndex(0),
    fRunLastStepIndex(0),
    fEventComponents(),
    fActiveEventComponents(),
    fEventIndex(0),
    fEventFirstTrackIndex(0),
    fEventLastTrackIndex(0),
    fEventFirstStepIndex(0),
    fEventLastStepIndex(0),
    fTrackComponents(),
    fActiveTrackComponents(),
    fTrackIndex(0),
    fTrackFirstStepIndex(0),
    fTrackLastStepIndex(0),
    fStepComponent(false),
    fStepComponents(),
    fActiveStepComponents(),
    fStepIndex(0)
{
    fPrecision = aCopy.Precision();
}

KSWriteASCII* KSWriteASCII::Clone() const
{
    return new KSWriteASCII(*this);
}

KSWriteASCII::~KSWriteASCII() = default;

void KSWriteASCII::ExecuteRun()
{
    wtrmsg_debug("ASCII writer <" << GetName() << "> is filling a run" << eom);

    if (fEventIndex != 0)
        fRunLastEventIndex = fEventIndex - 1;

    if (fTrackIndex != 0)
        fRunLastTrackIndex = fTrackIndex - 1;

    if (fStepIndex != 0)
        fRunLastStepIndex = fStepIndex - 1;

    for (auto& activeRunComponent : fActiveRunComponents)
        activeRunComponent.second->Fill();

    fRunIndex++;
    fRunFirstEventIndex = fEventIndex;
    fRunFirstTrackIndex = fTrackIndex;
    fRunFirstStepIndex = fStepIndex;

    return;
}

void KSWriteASCII::ExecuteEvent()
{
    wtrmsg_debug("ASCII writer <" << GetName() << "> is filling an event" << eom);

    if (fTrackIndex != 0)
        fEventLastTrackIndex = fTrackIndex - 1;

    if (fStepIndex != 0)
        fEventLastStepIndex = fStepIndex - 1;

    for (auto& activeEventComponent : fActiveEventComponents)
        activeEventComponent.second->Fill();

    fEventIndex++;
    fEventFirstTrackIndex = fTrackIndex;
    fEventFirstStepIndex = fStepIndex;

    return;
}
void KSWriteASCII::ExecuteTrack()
{
    wtrmsg_debug("ASCII writer <" << GetName() << "> is filling a track" << eom);

    if (fStepIndex != 0)
        fTrackLastStepIndex = fStepIndex - 1;

    fTextFile->File()->put('\n');
    for (auto& activeTrackComponent : fActiveTrackComponents)
        activeTrackComponent.second->Fill();

    wtrmsg(eNormal) << "ASCII output was written to file <" << fTextFile->GetName() << ">" << eom;
    fTextFile->Close();

    delete fTextFile;

    // new output file

    fTextFile = MakeOutputFile(fTrackIndex + 1);
    if (fTextFile->Open(KFile::eWrite) == true) {
        // do nothing here
    }
    else {
        wtrmsg(eError) << "could not open ASCII output file" << eom;
    }

    ComponentIt tIt;

    for (tIt = fTrackComponents.begin(); tIt != fTrackComponents.end(); tIt++)
        tIt->second->MakeTitle(tIt->first, 0);

    fTextFile->File()->put('\n');
    for (tIt = fStepComponents.begin(); tIt != fStepComponents.end(); tIt++)
        tIt->second->MakeTitle(tIt->first, 1);

    fTrackIndex++;
    fTrackFirstStepIndex = fStepIndex;

    return;
}
void KSWriteASCII::ExecuteStep()
{
    if (fStepIterationIndex % fStepIteration != 0) {
        wtrmsg_debug("ASCII writer <" << GetName() << "> is skipping a step because of step iteration value <"
                                      << fStepIteration << ">" << eom);
        fStepIterationIndex++;
        return;
    }

    fTextFile->File()->put('\n');
    if (fStepComponent == true) {
        wtrmsg_debug("ASCII writer <" << GetName() << "> is filling a step" << eom);

        for (auto& activeStepComponent : fActiveStepComponents)
            activeStepComponent.second->Fill();
    }

    fStepIndex++;
    fStepIterationIndex++;

    return;
}

void KSWriteASCII::AddRunComponent(KSComponent* aComponent)
{
    auto tIt = fRunComponents.find(aComponent);
    if (tIt == fRunComponents.end()) {
        wtrmsg_debug("ASCII writer is making a new run output called <" << aComponent->GetName() << ">" << eom);


        fKey = aComponent->GetName();


        auto* tRunData = new Data(aComponent, this);
        tIt = fRunComponents.insert(ComponentEntry(aComponent, tRunData)).first;
    }

    wtrmsg_debug("ASCII writer is starting a run output called <" << aComponent->GetName() << ">" << eom);

    tIt->second->Start(fRunIndex);
    fActiveRunComponents.insert(*tIt);

    return;
}

void KSWriteASCII::RemoveRunComponent(KSComponent* aComponent)
{
    auto tIt = fActiveRunComponents.find(aComponent);
    if (tIt == fActiveRunComponents.end()) {
        wtrmsg(eError) << "ASCII writer has no run output called <" << aComponent->GetName() << ">" << eom;
    }

    fActiveRunComponents.erase(tIt);

    return;
}

void KSWriteASCII::AddEventComponent(KSComponent* aComponent)
{
    auto tIt = fEventComponents.find(aComponent);
    if (tIt == fEventComponents.end()) {
        wtrmsg_debug("ASCII writer is making a new event output called <" << aComponent->GetName() << ">" << eom);

        fKey = aComponent->GetName();

        auto* tEventData = new Data(aComponent, this);
        tIt = fEventComponents.insert(ComponentEntry(aComponent, tEventData)).first;
    }

    wtrmsg_debug("ASCII writer is starting a event output called <" << aComponent->GetName() << ">" << eom);

    tIt->second->Start(fEventIndex);
    fActiveEventComponents.insert(*tIt);

    return;
}

void KSWriteASCII::RemoveEventComponent(KSComponent* aComponent)
{
    auto tIt = fActiveEventComponents.find(aComponent);
    if (tIt == fActiveEventComponents.end()) {
        wtrmsg(eError) << "ASCII writer has no event output called <" << aComponent->GetName() << ">" << eom;
    }

    fActiveEventComponents.erase(tIt);

    return;
}

void KSWriteASCII::AddTrackComponent(KSComponent* aComponent)
{

    auto tIt = fTrackComponents.find(aComponent);
    if (tIt == fTrackComponents.end()) {
        wtrmsg_debug("ASCII writer is making a new track output called <" << aComponent->GetName() << ">" << eom);

        fKey = aComponent->GetName();

        auto* tTrackData = new Data(aComponent, this);
        tIt = fTrackComponents.insert(ComponentEntry(aComponent, tTrackData)).first;
    }

    wtrmsg_debug("ASCII writer is starting a track output called <" << aComponent->GetName() << ">" << eom);

    tIt->second->Start(fTrackIndex);
    fActiveTrackComponents.insert(*tIt);

    return;
}

void KSWriteASCII::RemoveTrackComponent(KSComponent* aComponent)
{
    auto tIt = fActiveTrackComponents.find(aComponent);
    if (tIt == fActiveTrackComponents.end()) {
        wtrmsg(eError) << "ASCII writer has no track output called <" << aComponent->GetName() << ">" << eom;
    }

    fActiveTrackComponents.erase(tIt);

    return;
}

void KSWriteASCII::AddStepComponent(KSComponent* aComponent)
{
    if (fStepComponent == false)
        fStepComponent = true;

    auto tIt = fStepComponents.find(aComponent);
    if (tIt == fStepComponents.end()) {
        wtrmsg_debug("ASCII writer is making a new step output called <" << aComponent->GetName() << ">" << eom);

        fKey = aComponent->GetName();

        auto* tStepData = new Data(aComponent, this);

        tIt = fStepComponents.insert(ComponentEntry(aComponent, tStepData)).first;
    }

    wtrmsg_debug("ASCII writer is starting a step output called <" << aComponent->GetName() << ">" << eom);

    tIt->second->Start(fStepIndex);

    fActiveStepComponents.insert(*tIt);

    return;
}

void KSWriteASCII::RemoveStepComponent(KSComponent* aComponent)
{
    auto tIt = fActiveStepComponents.find(aComponent);
    if (tIt == fActiveStepComponents.end()) {
        wtrmsg(eError) << "ASCII writer has no step output called <" << aComponent->GetName() << ">" << eom;
    }

    fActiveStepComponents.erase(tIt);

    return;
}

void KSWriteASCII::InitializeComponent()
{
    wtrmsg_debug("starting ASCII writer" << eom);

    fTextFile = MakeOutputFile(fTrackIndex);

    if (fTextFile->Open(KFile::eWrite) == true) {
        // do nothing here
    }
    else {
        wtrmsg(eError) << "could not open ASCII output file" << eom;
    }

    return;
}

void KSWriteASCII::DeinitializeComponent()

{
    ComponentIt tIt;

    for (tIt = fTrackComponents.begin(); tIt != fTrackComponents.end(); tIt++)
        delete tIt->second;

    for (tIt = fStepComponents.begin(); tIt != fStepComponents.end(); tIt++)
        delete tIt->second;

    fTextFile->Close();

    delete fTextFile;

    return;
}

KTextFile* KSWriteASCII::MakeOutputFile(int anIndex) const
{
    stringstream s;
    s << fBase << "_Track" << anIndex << +".txt";
    string tBase = s.str();
    string tPath = fPath.empty() ? OUTPUT_DEFAULT_DIR : fPath;

#ifdef Kassiopeia_USE_BOOST
    KPathUtils::MakeDirectory(tPath);
#endif
    return KTextFile::CreateOutputTextFile(tPath, tBase);
}

STATICINT sKSWriteASCIIDict =
    KSDictionary<KSWriteASCII>::AddCommand(&KSWriteASCII::AddRunComponent, &KSWriteASCII::RemoveRunComponent,
                                           "add_run_output", "remove_run_output") +
    KSDictionary<KSWriteASCII>::AddCommand(&KSWriteASCII::AddEventComponent, &KSWriteASCII::RemoveEventComponent,
                                           "add_event_output", "remove_event_output") +
    KSDictionary<KSWriteASCII>::AddCommand(&KSWriteASCII::AddTrackComponent, &KSWriteASCII::RemoveTrackComponent,
                                           "add_track_output", "remove_track_output") +
    KSDictionary<KSWriteASCII>::AddCommand(&KSWriteASCII::AddStepComponent, &KSWriteASCII::RemoveStepComponent,
                                           "add_step_output", "remove_step_output");
}  // namespace Kassiopeia
