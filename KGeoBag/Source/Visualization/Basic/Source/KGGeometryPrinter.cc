#include "KGGeometryPrinter.hh"

#include "KGVisualizationMessage.hh"

#include <boost/optional.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
namespace pt = boost::property_tree;
namespace boost
{
namespace property_tree
{
typedef ptree::path_type path;
using value = ptree::value_type;
}  // namespace property_tree
};  // namespace boost

#include "KConst.h"
#include "KFile.h"
#include "KRotation.hh"
#include "KBaseStringUtils.h"

#include <cmath>

#undef NDEBUG
#include <cassert>

using namespace std;
using namespace katrin;

using katrin::KThreeVector;
using katrin::KTwoVector;

namespace KGeoBag
{

class KGGeometryPrinter::Private
{
  public:
    Private() = default;
    ;

    inline string Put(const string& aKey)
    {
        auto tPath = pt::path(aKey, '/');
        if (!fData.get_child_optional(tPath))
            fData.put_child(tPath, pt::ptree());
        return tPath.dump();
    }
    template<typename T> inline string Put(const string& aKey, const T& aValue)
    {
        auto tPath = pt::path(aKey, '/');
        fData.put(tPath, aValue);
        return tPath.dump();
    }
    inline string Put(const string& aKey, const KTwoVector& aValue)
    {
        stringstream ss;
        ss << aValue.X() << " " << aValue.Y();
        return Put(aKey, ss.str());
    }
    inline string Put(const string& aKey, const KThreeVector& aValue)
    {
        stringstream ss;
        ss << aValue.X() << " " << aValue.Y() << " " << aValue.Z();
        return Put(aKey, ss.str());
    }
    inline string Put(const string& aKey, const set<string>& aValue)
    {
        return Put(aKey, KBaseStringUtils::Join(aValue, " "));
    }
    inline string Put(const string& aKey, const vector<string>& aValue)
    {
        return Put(aKey, KBaseStringUtils::Join(aValue, " "));
    }

    template<typename T> inline const T& Get(const string& aKey)
    {
        auto tPath = pt::path(aKey, '/');
        return fData.get<T>(tPath);
    }

    template<typename T> inline const T& Get(const string& aKey, const T& aDefaultValue)
    {
        auto tPath = pt::path(aKey, '/');
        auto tValue = fData.get_optional<T>(tPath);
        return tValue ? tValue : aDefaultValue;
    }

    enum GeometryPrintFormat
    {
        eIniFormat,
        eJsonFormat,
        eXmlFormat
    };

    void Dump(std::ostream& aStream = std::cout, GeometryPrintFormat aFormat = eXmlFormat)
    {
        switch (aFormat) {
            case eIniFormat:
                pt::write_ini(aStream, fData);
                break;

            case eJsonFormat:
                pt::write_json(aStream, fData, true);
                break;

            case eXmlFormat:
                pt::xml_writer_settings<string> tSettings(' ', 4, "utf-8");
                pt::write_xml(aStream, fData, tSettings);
                break;
        };
    }

  protected:
    pt::ptree fData;

    friend class KGGeometryPrinter;
};

KGGeometryPrinter::KGGeometryPrinter() :
    fPath(""),
    fWriteJSON(false),
    fWriteXML(false),
    fWriteDOT(false),
    fUseColors(true),
    fStream(&std::cout),
    fPrivate(new Private()),
    fCurrentSpace(nullptr),
    fCurrentSurface(nullptr),
    fCurrentOrigin(KThreeVector::sZero),
    fCurrentXAxis(KThreeVector::sXUnit),
    fCurrentYAxis(KThreeVector::sYUnit),
    fCurrentZAxis(KThreeVector::sZUnit),
    fIgnore(true)
{}
KGGeometryPrinter::~KGGeometryPrinter() = default;

void KGGeometryPrinter::Render()
{
    fPrivate->fData.clear();

    vismsg(eInfo) << "geometry printer processing " << fSurfaces.size() << " surfaces" << eom;
    for (auto& tSurface : fSurfaces) {
        tSurface->AcceptNode(this);
    }

    vismsg(eInfo) << "geometry printer processing " << fSpaces.size() << " spaces" << eom;
    for (auto& tSpace : fSpaces) {
        tSpace->AcceptNode(this);
    }

    return;
}
void KGGeometryPrinter::Display()
{
    fPrivate->Dump(*fStream);

    return;
}
void KGGeometryPrinter::Write()
{
    string tFileBase;

    if (fFile.length() > 0) {
        if (!fPath.empty()) {
            tFileBase = string(fPath) + string("/") + fFile;
        }
        else {
            tFileBase = string(OUTPUT_DEFAULT_DIR) + string("/") + fFile;
        }
    }
    else {
        if (!fPath.empty()) {
            tFileBase = string(fPath) + string("/") + GetName();
        }
        else {
            tFileBase = string(OUTPUT_DEFAULT_DIR) + string("/") + GetName();
        }
    }

    if (fWriteJSON) {
        string tFileName = tFileBase + ".json";
        vismsg(eInfo) << "geometry printer writing to file <" << tFileName << ">" << eom;

        ofstream tFileStream(tFileName);
        fPrivate->Dump(tFileStream, Private::eJsonFormat);
        tFileStream.close();
    }

    if (fWriteXML) {
        string tFileName = tFileBase + ".xml";
        vismsg(eInfo) << "geometry printer writing to file <" << tFileName << ">" << eom;

        ofstream tFileStream(tFileName);
        fPrivate->Dump(tFileStream, Private::eXmlFormat);
        tFileStream.close();
    }

    if (fWriteDOT) {
        string tFileName = tFileBase + ".dot";
        vismsg(eInfo) << "geometry printer writing to file <" << tFileName << ">" << eom;

        ofstream tFileStream(tFileName);
        WriteGraphViz(tFileStream);
        tFileStream.close();
    }
    return;
}

void KGGeometryPrinter::WriteGraphViz(std::ostream& aStream, bool with_tags) const
{
    vector<std::string> edges;
    vector<std::string> nodes;

    for (auto &node : fVisitedSpaces) {
        std::string node_label = node->GetName();
        if (with_tags) {
            for (auto &tag : node->GetTags())
                node_label += " @" + tag;
        }
        std::string node_txt = "\"" + node->GetPath() + "\" [label=\"" + node_label + "\"; shape=ellipse;]";
        if (std::find(nodes.begin(), nodes.end(), node_txt) == nodes.end())
            nodes.push_back(node_txt);

        if (node->GetParent()) {
            std::string edge_txt = "\"" + node->GetParent()->GetPath() + "\" -> \"" + node->GetPath() + "\"";
            if (std::find(edges.begin(), edges.end(), edge_txt) == edges.end())
                edges.push_back(edge_txt);
        }
    }

    for (auto &node : fVisitedSurfaces) {
        std::string node_label = node->GetName();
        if (with_tags) {
            for (auto &tag : node->GetTags())
                node_label += " @" + tag;
        }
        std::string node_txt = "\"" + node->GetPath() + "\" [label=\"" + node_label + "\"; shape=box;]";
        if (std::find(nodes.begin(), nodes.end(), node_txt) == nodes.end())
            nodes.push_back(node_txt);

        if (node->GetParent()) {
            std::string edge_txt = "\"" + node->GetParent()->GetPath() + "\" -> \"" + node->GetPath() + "\"";
            if (std::find(edges.begin(), edges.end(), edge_txt) == edges.end())
                edges.push_back(edge_txt);
        }
    }

    aStream << "digraph G {" << endl;
    aStream << endl;
    for (auto & it : nodes)
        aStream << "\t" << it << endl;
    aStream << endl;
    for (auto & it : edges)
        aStream << "\t" << it << endl;
    aStream << endl;
    aStream << "}" << endl;
}

void KGGeometryPrinter::SetFile(const string& aFile)
{
    fFile = aFile;
    return;
}
const string& KGGeometryPrinter::GetFile() const
{
    return fFile;
}
void KGGeometryPrinter::SetPath(const string& aPath)
{
    fPath = aPath;
    return;
}
void KGGeometryPrinter::SetStream(std::ostream& aStream)
{
    fStream = &aStream;
}
void KGGeometryPrinter::SetUseColors(bool aFlag)
{
    fUseColors = aFlag;
}
void KGGeometryPrinter::SetWriteJSON(bool aFlag)
{
    fWriteJSON = aFlag;
    return;
}
void KGGeometryPrinter::SetWriteXML(bool aFlag)
{
    fWriteXML = aFlag;
    return;
}
void KGGeometryPrinter::SetWriteDOT(bool aFlag)
{
    fWriteDOT = aFlag;
    return;
}

void KGGeometryPrinter::AddSurface(KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}
void KGGeometryPrinter::AddSpace(KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}
/*
template<typename KeyT>
std::string KGGeometryPrinter::ColorHash(const KeyT& aValue)
{
#if 0
    // RGB COLOR in range 0x404040 .. 0xBFBFBF
    //size_t hash = floor((std::hash<KeyT>{}(aValue) / (long double)SIZE_MAX) * 0xFFFFFF);
    size_t hash = floor((std::hash<KeyT>{}(aValue) / (long double)SIZE_MAX) * 0x7F7F7F) + 0x404040;

    int r = (hash & (0xFF0000) ) >> 16;
    int g = (hash & (0x00FF00) ) >> 8;
    int b = (hash & (0x0000FF) ) >> 0;

    std::stringstream ss;
    ss << "\033[38;2;" << r << ";" << g << ";" << b << "m";
    return ss.str();
#else
    // ANSI COLOR in range 17 .. 230
    // color 0..15 is main
    // color 16 is black
    // color 231 is white
    // color 232..255 is grayscale
    size_t hash = floor((std::hash<KeyT>{}(aValue) / (long double)SIZE_MAX) * 213) + 17;

    std::stringstream ss;
    ss << "\033[38;5;" << hash <<  "m";
    return ss.str();
#endif
}

template<>
std::string KGGeometryPrinter::ColorHash(const double& aValue)
{
   return ColorHash(std::to_string(aValue));
}

template<>
std::string KGGeometryPrinter::ColorHash(const int& aValue)
{
   return ColorHash(std::to_string(aValue));
}

template <template <typename, typename...> class Container, typename T>
std::string KGGeometryPrinter::ColorizeAndJoin(const Container<T>& aSequence, std::string aSeparator)
{
    std::stringstream ss;
    ss << "<";
    for (auto it = aSequence.begin(); it != aSequence.end(); ++it) {
        ss << (it == aSequence.begin() ? "" : aSeparator);
        if (!fUseColors)
            ss << (*it);
        else
            ss << ColorHash(*it) << (*it) << "\033[0m";
    }
    ss << ">";
    return ss.str();
}

std::string KGGeometryPrinter::Colorize(const double& aValue)
{
    return Colorize(std::to_string(aValue));
}

std::string KGGeometryPrinter::Colorize(const KTwoVector& aValue)
{
    return ColorizeAndJoin(aValue.ComponentVector());
}

std::string KGGeometryPrinter::Colorize(const KThreeVector& aValue)
{
    return ColorizeAndJoin(aValue.ComponentVector());
}

std::string KGGeometryPrinter::Colorize(const std::string& aValue)
{
    if (!fUseColors)
        return aValue;
    return ColorHash(aValue) + aValue + "\033[0m";
}
*/

//****************
//surface visitors
//****************

void KGGeometryPrinter::VisitSurface(KGSurface* aSurface)
{
    const KGSpace* tParent = aSurface->GetParent();
    if (tParent) {
        VisitSpace(const_cast<KGSpace*>(tParent));
    }

    fVisitedSurfaces.push_back(aSurface);

    string tRoot = fPrivate->Put(aSurface->GetPath());
    fPrivate->Put(tRoot + "/type", aSurface->Name());
    fPrivate->Put(tRoot + "/name", aSurface->GetName());
    fPrivate->Put(tRoot + "/path", aSurface->GetPath());
    fPrivate->Put(tRoot + "/tags", aSurface->GetTags());

    fCurrentSurface = aSurface;
    fCurrentOrigin = aSurface->GetOrigin();
    fCurrentXAxis = aSurface->GetXAxis();
    fCurrentYAxis = aSurface->GetYAxis();
    fCurrentZAxis = aSurface->GetZAxis();

    KRotation tRotation;
    tRotation.SetRotatedFrame(fCurrentXAxis, fCurrentYAxis, fCurrentZAxis);

    fPrivate->Put(tRoot + "/system/origin", fCurrentOrigin);
    fPrivate->Put(tRoot + "/system/x_axis", fCurrentXAxis);
    fPrivate->Put(tRoot + "/system/y_axis", fCurrentYAxis);
    fPrivate->Put(tRoot + "/system/z_axis", fCurrentZAxis);

    double tAlpha, tBeta, tGamma;
    tRotation.GetEulerAnglesInDegrees(tAlpha, tBeta, tGamma);

    fPrivate->Put(tRoot + "/system/alpha", tAlpha);
    fPrivate->Put(tRoot + "/system/beta", tBeta);
    fPrivate->Put(tRoot + "/system/gamma", tGamma);

    fIgnore = false;

    return;
}
void KGGeometryPrinter::VisitAnnulusSurface(KGAnnulusSurface* aSurface)
{
    KThreeVector P1;
    LocalToGlobal(KThreeVector(0, 0, aSurface->Z()), P1);

    double tArea1 = KConst::Pi() * aSurface->R1() * aSurface->R1();
    double tArea2 = KConst::Pi() * aSurface->R2() * aSurface->R2();

    string tPath = fCurrentSurface->GetPath() + "/" + aSurface->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSurface->Name());
    fPrivate->Put(tRoot + "/name", aSurface->GetName());
    fPrivate->Put(tRoot + "/tags", aSurface->GetTags());

    fPrivate->Put(tRoot + "/source_object/z", aSurface->Z());
    fPrivate->Put(tRoot + "/source_object/r1", aSurface->R1());
    fPrivate->Put(tRoot + "/source_object/r2", aSurface->R2());
    fPrivate->Put(tRoot + "/source_object/width", aSurface->R2() - aSurface->R1());
    fPrivate->Put(tRoot + "/source_object/area", tArea2 - tArea1);

    fPrivate->Put(tRoot + "/global_coords/point1", P1);

    //clear surface
    fCurrentSurface = nullptr;
}
void KGGeometryPrinter::VisitConeSurface(KGConeSurface* aSurface)
{
    KThreeVector P1, P2;
    LocalToGlobal(KThreeVector(0, 0, aSurface->ZA()), P1);
    LocalToGlobal(KThreeVector(0, 0, aSurface->ZB()), P2);

    KThreeVector N = P2 - P1;
    double tLength = fabs(aSurface->ZB() - aSurface->ZA());
    double tSlantHeight = sqrt(aSurface->RB() * aSurface->RB() + tLength * tLength);
    double tArea = KConst::Pi() * aSurface->RB() * tSlantHeight + KConst::Pi() * aSurface->RB() * aSurface->RB();

    string tPath = fCurrentSurface->GetPath() + "/" + aSurface->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSurface->Name());
    fPrivate->Put(tRoot + "/name", aSurface->GetName());
    fPrivate->Put(tRoot + "/tags", aSurface->GetTags());

    fPrivate->Put(tRoot + "/source_object/z1", aSurface->ZA());
    fPrivate->Put(tRoot + "/source_object/z2", aSurface->ZB());
    fPrivate->Put(tRoot + "/source_object/r2", aSurface->RB());
    fPrivate->Put(tRoot + "/source_object/length", tLength);
    fPrivate->Put(tRoot + "/source_object/area", tArea);

    fPrivate->Put(tRoot + "/global_coords/point1", P1);
    fPrivate->Put(tRoot + "/global_coords/point2", P2);
    fPrivate->Put(tRoot + "/global_coords/normal", N);

    //clear surface
    fCurrentSurface = nullptr;
}
void KGGeometryPrinter::VisitCutConeSurface(KGCutConeSurface* aSurface)
{
    KThreeVector P1, P2;
    LocalToGlobal(KThreeVector(0, 0, aSurface->Z1()), P1);
    LocalToGlobal(KThreeVector(0, 0, aSurface->Z2()), P2);

    KThreeVector N = P2 - P1;
    double tLength = fabs(aSurface->Z2() - aSurface->Z1());
    double tSlantHeight = sqrt(aSurface->R1() * aSurface->R1() - aSurface->R1() * aSurface->R2() +
                               aSurface->R2() * aSurface->R2() + tLength * tLength);
    double tArea = KConst::Pi() * (aSurface->R1() + aSurface->R2()) * tSlantHeight +
                   KConst::Pi() * (aSurface->R1() * aSurface->R1() + aSurface->R2() * aSurface->R2());

    string tPath = fCurrentSurface->GetPath() + "/" + aSurface->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSurface->Name());
    fPrivate->Put(tRoot + "/name", aSurface->GetName());
    fPrivate->Put(tRoot + "/tags", aSurface->GetTags());

    fPrivate->Put(tRoot + "/source_object/z1", aSurface->Z1());
    fPrivate->Put(tRoot + "/source_object/z2", aSurface->Z2());
    fPrivate->Put(tRoot + "/source_object/r1", aSurface->R1());
    fPrivate->Put(tRoot + "/source_object/r2", aSurface->R2());
    fPrivate->Put(tRoot + "/source_object/length", tLength);
    fPrivate->Put(tRoot + "/source_object/area", tArea);

    fPrivate->Put(tRoot + "/global_coords/point1", P1);
    fPrivate->Put(tRoot + "/global_coords/point2", P2);
    fPrivate->Put(tRoot + "/global_coords/normal", N);

    //clear surface
    fCurrentSurface = nullptr;
}
void KGGeometryPrinter::VisitCylinderSurface(KGCylinderSurface* aSurface)
{
    KThreeVector P1, P2;
    LocalToGlobal(KThreeVector(0, 0, aSurface->Z1()), P1);
    LocalToGlobal(KThreeVector(0, 0, aSurface->Z2()), P2);

    KThreeVector N = P2 - P1;
    double tLength = fabs(aSurface->Z2() - aSurface->Z1());
    double tArea = 2 * KConst::Pi() * tLength * aSurface->R();

    string tPath = fCurrentSurface->GetPath() + "/" + aSurface->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSurface->Name());
    fPrivate->Put(tRoot + "/name", aSurface->GetName());
    fPrivate->Put(tRoot + "/tags", aSurface->GetTags());

    fPrivate->Put(tRoot + "/source_object/z1", aSurface->Z1());
    fPrivate->Put(tRoot + "/source_object/z2", aSurface->Z2());
    fPrivate->Put(tRoot + "/source_object/r", aSurface->R());
    fPrivate->Put(tRoot + "/source_object/length", tLength);
    fPrivate->Put(tRoot + "/source_object/area", tArea);

    fPrivate->Put(tRoot + "/global_coords/point1", P1);
    fPrivate->Put(tRoot + "/global_coords/point2", P2);
    fPrivate->Put(tRoot + "/global_coords/normal", N);

    //clear surface
    fCurrentSurface = nullptr;
}
void KGGeometryPrinter::VisitDiskSurface(KGDiskSurface* aSurface)
{
    KThreeVector P1;
    LocalToGlobal(KThreeVector(0, 0, aSurface->Z()), P1);

    double tArea = KConst::Pi() * aSurface->R() * aSurface->R();

    string tPath = fCurrentSurface->GetPath() + "/" + aSurface->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSurface->Name());
    fPrivate->Put(tRoot + "/name", aSurface->GetName());
    fPrivate->Put(tRoot + "/tags", aSurface->GetTags());

    fPrivate->Put(tRoot + "/source_object/z", aSurface->Z());
    fPrivate->Put(tRoot + "/source_object/r", aSurface->R());
    fPrivate->Put(tRoot + "/source_object/area", tArea);

    fPrivate->Put(tRoot + "/global_coords/point1", P1);

    //clear surface
    fCurrentSurface = nullptr;
}
void KGGeometryPrinter::VisitWrappedSurface(KGRodSurface* aSurface)
{
    auto tRod = aSurface->GetObject();

    vector<KThreeVector> Pi(tRod->GetNCoordinates());
    for (unsigned i = 0; i < Pi.size(); i++) {
        LocalToGlobal(tRod->GetCoordinate(i), Pi[i]);
    }

    double tArea = 2. * KConst::Pi() * tRod->GetRadius() * tRod->GetLength();

    string tPath = fCurrentSurface->GetPath() + "/" + aSurface->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSurface->Name());
    fPrivate->Put(tRoot + "/name", aSurface->GetName());
    fPrivate->Put(tRoot + "/tags", aSurface->GetTags());

    fPrivate->Put(tRoot + "/source_object/r", tRod->GetRadius());
    fPrivate->Put(tRoot + "/source_object/length", tRod->GetLength());
    fPrivate->Put(tRoot + "/source_object/area", tArea);

    for (unsigned i = 0; i < Pi.size(); i++) {
        fPrivate->Put(tRoot + "/global/point" + std::to_string(i), Pi[i]);
    }

    if (Pi.size() == 2) {
        KThreeVector N = Pi[1] - Pi[0];
        fPrivate->Put(tRoot + "/global/normal", N);
    }

    //clear space
    fCurrentSpace = nullptr;
}

//**************
//space visitors
//**************

void KGGeometryPrinter::VisitSpace(KGSpace* aSpace)
{
    const KGSpace* tParent = aSpace->GetParent();
    if (tParent && tParent != KGInterface::GetInstance()->Root()) {
        VisitSpace(const_cast<KGSpace*>(tParent));
    }

    fVisitedSpaces.push_back(aSpace);

    string tRoot = fPrivate->Put(aSpace->GetPath());
    fPrivate->Put(tRoot + "/type", aSpace->Name());
    fPrivate->Put(tRoot + "/name", aSpace->GetName());
    fPrivate->Put(tRoot + "/path", aSpace->GetPath());
    fPrivate->Put(tRoot + "/tags", aSpace->GetTags());

    fCurrentSpace = aSpace;
    fCurrentOrigin = aSpace->GetOrigin();
    fCurrentXAxis = aSpace->GetXAxis();
    fCurrentYAxis = aSpace->GetYAxis();
    fCurrentZAxis = aSpace->GetZAxis();

    KRotation tRotation;
    tRotation.SetRotatedFrame(fCurrentXAxis, fCurrentYAxis, fCurrentZAxis);

    fPrivate->Put(tRoot + "/system/origin", fCurrentOrigin);
    fPrivate->Put(tRoot + "/system/x_axis", fCurrentXAxis);
    fPrivate->Put(tRoot + "/system/y_axis", fCurrentYAxis);
    fPrivate->Put(tRoot + "/system/z_axis", fCurrentZAxis);

    double tAlpha, tBeta, tGamma;
    tRotation.GetEulerAnglesInDegrees(tAlpha, tBeta, tGamma);

    fPrivate->Put(tRoot + "/system/alpha", tAlpha);
    fPrivate->Put(tRoot + "/system/beta", tBeta);
    fPrivate->Put(tRoot + "/system/gamma", tGamma);

    fIgnore = false;

    return;
}

void KGGeometryPrinter::VisitConeSpace(KGConeSpace* aSpace)
{
    KThreeVector P1, P2;
    LocalToGlobal(KThreeVector(0, 0, aSpace->ZA()), P1);
    LocalToGlobal(KThreeVector(0, 0, aSpace->ZB()), P2);

    KThreeVector N = P2 - P1;
    double tLength = fabs(aSpace->ZB() - aSpace->ZA());
    double tVolume = KConst::Pi() * tLength / 3. * aSpace->RB();

    string tPath = fCurrentSpace->GetPath() + "/" + aSpace->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSpace->Name());
    fPrivate->Put(tRoot + "/name", aSpace->GetName());
    fPrivate->Put(tRoot + "/tags", aSpace->GetTags());

    fPrivate->Put(tRoot + "/source_object/z1", aSpace->ZA());
    fPrivate->Put(tRoot + "/source_object/z2", aSpace->ZB());
    fPrivate->Put(tRoot + "/source_object/r2", aSpace->RB());
    fPrivate->Put(tRoot + "/source_object/length", tLength);
    fPrivate->Put(tRoot + "/source_object/volume", tVolume);

    fPrivate->Put(tRoot + "/global_coords/point1", P1);
    fPrivate->Put(tRoot + "/global_coords/point2", P2);
    fPrivate->Put(tRoot + "/global_coords/normal", N);

    //clear surface
    fCurrentSpace = nullptr;
}
void KGGeometryPrinter::VisitCutConeSpace(KGCutConeSpace* aSpace)
{
    KThreeVector P1, P2;
    LocalToGlobal(KThreeVector(0, 0, aSpace->Z1()), P1);
    LocalToGlobal(KThreeVector(0, 0, aSpace->Z2()), P2);

    KThreeVector N = P2 - P1;
    double tLength = fabs(aSpace->Z2() - aSpace->Z1());
    double tVolume = KConst::Pi() * tLength / 3. *
                     (aSpace->R1() * aSpace->R1() + aSpace->R1() * aSpace->R2() + aSpace->R2() * aSpace->R2());

    string tPath = fCurrentSpace->GetPath() + "/" + aSpace->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSpace->Name());
    fPrivate->Put(tRoot + "/name", aSpace->GetName());
    fPrivate->Put(tRoot + "/tags", aSpace->GetTags());

    fPrivate->Put(tRoot + "/source_object/z1", aSpace->Z1());
    fPrivate->Put(tRoot + "/source_object/z2", aSpace->Z2());
    fPrivate->Put(tRoot + "/source_object/r1", aSpace->R1());
    fPrivate->Put(tRoot + "/source_object/r2", aSpace->R2());
    fPrivate->Put(tRoot + "/source_object/length", tLength);
    fPrivate->Put(tRoot + "/source_object/volume", tVolume);

    fPrivate->Put(tRoot + "/global_coords/point1", P1);
    fPrivate->Put(tRoot + "/global_coords/point2", P2);
    fPrivate->Put(tRoot + "/global_coords/normal", N);

    //clear space
    fCurrentSpace = nullptr;
}
void KGGeometryPrinter::VisitCutConeTubeSpace(KGCutConeTubeSpace* aSpace)
{
    KThreeVector P1, P2;
    LocalToGlobal(KThreeVector(0, 0, aSpace->Z1()), P1);
    LocalToGlobal(KThreeVector(0, 0, aSpace->Z2()), P2);

    KThreeVector N = P2 - P1;
    double tLength = fabs(aSpace->Z2() - aSpace->Z1());
    double tVolume1 = KConst::Pi() * tLength / 3. *
                      (aSpace->R11() * aSpace->R11() + aSpace->R11() * aSpace->R12() + aSpace->R12() * aSpace->R12());
    double tVolume2 = KConst::Pi() * tLength / 3. *
                      (aSpace->R21() * aSpace->R21() + aSpace->R21() * aSpace->R22() + aSpace->R22() * aSpace->R22());

    string tPath = fCurrentSpace->GetPath() + "/" + aSpace->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSpace->Name());
    fPrivate->Put(tRoot + "/name", aSpace->GetName());
    fPrivate->Put(tRoot + "/tags", aSpace->GetTags());

    fPrivate->Put(tRoot + "/source_object/z1", aSpace->Z1());
    fPrivate->Put(tRoot + "/source_object/z2", aSpace->Z2());
    fPrivate->Put(tRoot + "/source_object/r11", aSpace->R11());
    fPrivate->Put(tRoot + "/source_object/r12", aSpace->R12());
    fPrivate->Put(tRoot + "/source_object/r21", aSpace->R21());
    fPrivate->Put(tRoot + "/source_object/r22", aSpace->R22());
    fPrivate->Put(tRoot + "/source_object/length", tLength);
    fPrivate->Put(tRoot + "/source_object/volume", tVolume2 - tVolume1);
    fPrivate->Put(tRoot + "/source_object/width1", aSpace->R12() - aSpace->R11());
    fPrivate->Put(tRoot + "/source_object/width2", aSpace->R22() - aSpace->R21());

    fPrivate->Put(tRoot + "/global_coords/point1", P1);
    fPrivate->Put(tRoot + "/global_coords/point2", P2);
    fPrivate->Put(tRoot + "/global_coords/normal", N);

    //clear space
    fCurrentSpace = nullptr;
}
void KGGeometryPrinter::VisitCylinderSpace(KGCylinderSpace* aSpace)
{
    KThreeVector P1, P2;
    LocalToGlobal(KThreeVector(0, 0, aSpace->Z1()), P1);
    LocalToGlobal(KThreeVector(0, 0, aSpace->Z2()), P2);

    KThreeVector N = P2 - P1;
    double tLength = fabs(aSpace->Z2() - aSpace->Z1());
    double tVolume = KConst::Pi() * tLength * aSpace->R() * aSpace->R();

    string tPath = fCurrentSpace->GetPath() + "/" + aSpace->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSpace->Name());
    fPrivate->Put(tRoot + "/name", aSpace->GetName());
    fPrivate->Put(tRoot + "/tags", aSpace->GetTags());

    fPrivate->Put(tRoot + "/source_object/z1", aSpace->Z1());
    fPrivate->Put(tRoot + "/source_object/z2", aSpace->Z2());
    fPrivate->Put(tRoot + "/source_object/r", aSpace->R());
    fPrivate->Put(tRoot + "/source_object/length", tLength);
    fPrivate->Put(tRoot + "/source_object/volume", tVolume);

    fPrivate->Put(tRoot + "/global_coords/point1", P1);
    fPrivate->Put(tRoot + "/global_coords/point2", P2);
    fPrivate->Put(tRoot + "/global_coords/normal", N);

    //clear space
    fCurrentSpace = nullptr;
}
void KGGeometryPrinter::VisitCylinderTubeSpace(KGCylinderTubeSpace* aSpace)
{
    KThreeVector P1, P2;
    LocalToGlobal(KThreeVector(0, 0, aSpace->Z1()), P1);
    LocalToGlobal(KThreeVector(0, 0, aSpace->Z2()), P2);

    KThreeVector N = P2 - P1;
    double tLength = fabs(aSpace->Z2() - aSpace->Z1());
    double tVolume1 = KConst::Pi() * tLength * aSpace->R1() * aSpace->R1();
    double tVolume2 = KConst::Pi() * tLength * aSpace->R2() * aSpace->R2();

    string tPath = fCurrentSpace->GetPath() + "/" + aSpace->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSpace->Name());
    fPrivate->Put(tRoot + "/name", aSpace->GetName());
    fPrivate->Put(tRoot + "/tags", aSpace->GetTags());

    fPrivate->Put(tRoot + "/source_object/z1", aSpace->Z1());
    fPrivate->Put(tRoot + "/source_object/z2", aSpace->Z2());
    fPrivate->Put(tRoot + "/source_object/r1", aSpace->R1());
    fPrivate->Put(tRoot + "/source_object/r2", aSpace->R2());
    fPrivate->Put(tRoot + "/source_object/length", tLength);
    fPrivate->Put(tRoot + "/source_object/volume", tVolume2 - tVolume1);
    fPrivate->Put(tRoot + "/source_object/width", aSpace->R2() - aSpace->R1());

    fPrivate->Put(tRoot + "/global_coords/point1", P1);
    fPrivate->Put(tRoot + "/global_coords/point2", P2);
    fPrivate->Put(tRoot + "/global_coords/normal", N);

    //clear space
    fCurrentSpace = nullptr;
}
void KGGeometryPrinter::VisitWrappedSpace(KGRodSpace* aSpace)
{
    auto tRod = aSpace->GetObject();

    vector<KThreeVector> Pi(tRod->GetNCoordinates());
    for (unsigned i = 0; i < Pi.size(); i++) {
        LocalToGlobal(tRod->GetCoordinate(i), Pi[i]);
    }

    double tVolume = KConst::Pi() * tRod->GetRadius() * tRod->GetRadius() * tRod->GetLength();

    string tPath = fCurrentSpace->GetPath() + "/" + aSpace->GetName();
    string tRoot = fPrivate->Put(tPath);
    fPrivate->Put(tRoot + "/type", aSpace->Name());
    fPrivate->Put(tRoot + "/name", aSpace->GetName());
    fPrivate->Put(tRoot + "/tags", aSpace->GetTags());

    fPrivate->Put(tRoot + "/source_object/r", tRod->GetRadius());
    fPrivate->Put(tRoot + "/source_object/length", tRod->GetLength());
    fPrivate->Put(tRoot + "/source_object/volume", tVolume);

    for (unsigned i = 0; i < Pi.size(); i++) {
        fPrivate->Put(tRoot + "/global/point" + std::to_string(i), Pi[i]);
    }

    if (Pi.size() == 2) {
        KThreeVector N = Pi[1] - Pi[0];
        fPrivate->Put(tRoot + "/global/normal", N);
    }

    //clear space
    fCurrentSpace = nullptr;
}

void KGGeometryPrinter::LocalToGlobal(const KThreeVector& aLocal, KThreeVector& aGlobal)
{
    aGlobal = fCurrentOrigin + aLocal.X() * fCurrentXAxis + aLocal.Y() * fCurrentYAxis + aLocal.Z() * fCurrentZAxis;

    return;
}

}  // namespace KGeoBag
