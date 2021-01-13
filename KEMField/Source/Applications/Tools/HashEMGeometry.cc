#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"
#include "KMD5HashGenerator.hh"
#include "KSADataStreamer.hh"
#include "KSurfaceContainer.hh"
#include "KTypelist.hh"

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <sys/stat.h>

using namespace KEMField;

template<class Typelist> struct OmitAttributeInTypelist
{
    OmitAttributeInTypelist(KMD5HashGenerator& hashGenerator, const std::string& attributeName)
    {
        typedef typename Typelist::Head Head;
        using Tail = typename Typelist::Tail;

        if (Head::Name() == attributeName)
            hashGenerator.Omit(Type2Type<Head>());
        else
            OmitAttributeInTypelist<Tail>(hashGenerator, attributeName);
    }
};

template<> struct OmitAttributeInTypelist<KNullType>
{
    OmitAttributeInTypelist(KMD5HashGenerator& /*unused*/, const std::string& /*unused*/) {}
};

template<class Typelist> struct AppendNames
{
    AppendNames(const std::string& prefix, const std::string& suffix, std::string& usage)
    {
        using Head = typename Typelist::Head;
        using Tail = typename Typelist::Tail;

        usage.append(prefix);
        usage.append(Head::Name());
        usage.append(suffix);
        AppendNames<Tail>(prefix, suffix, usage);
    }
};

template<> struct AppendNames<KNullType>
{
    AppendNames(const std::string& /*unused*/, const std::string& /*unused*/, std::string& /*unused*/) {}
};

int main(int argc, char* argv[])
{
    std::string usage = "\n"
                        "Usage: HashEMGeometry <GeometryFile>\n"
                        "\n"
                        "This program takes a KEMField geometry file and generates a unique hash from it.\n"
                        "\n"
                        "\tAvailable options:\n"
                        "\t -h, --help               (shows this message and exits)\n"
                        "\t -n, --name               (name of surface container)\n"
                        "\t -o, --omit_attribute     (omits data for a surface attribute)\n"
                        "\t -m, --masked_bits        (masks the least significant n bits)\n"
                        "\n"
                        "\tOmittable attributes:\n";

    std::string prefix = "\t\t";
    std::string suffix = ",\n";

    AppendNames<KBasisTypes>(prefix, suffix, usage);
    AppendNames<KBoundaryTypes>(prefix, suffix, usage);
    AppendNames<KShapeTypes>(prefix, suffix, usage);

    usage = usage.substr(0, usage.find_last_of(suffix) - (suffix.size() - 1));
    usage.append("\n\n");

    if (argc == 1) {
        std::cout << usage;
        return 0;
    }

    static struct option longOptions[] = {
        {"help", no_argument, nullptr, 'h'},
        {"name", required_argument, nullptr, 'n'},
        {"masked_bits", required_argument, nullptr, 'm'},
        {"omit_attribute", required_argument, nullptr, 'o'},
    };

    static const char* optString = "hn:m:o:";

    KMD5HashGenerator hashGenerator;
    KSurfaceContainer surfaceContainer;
    std::string name = KSurfaceContainer::Name();

    while (true) {
        int optId = getopt_long(argc, argv, optString, longOptions, nullptr);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                return 0;
            case ('n'):
                name = optarg;
                break;
            case ('m'):
                hashGenerator.MaskedBits(atoi(optarg));
                break;
            case ('o'):
                OmitAttributeInTypelist<KBasisTypes>(hashGenerator, optarg);
                OmitAttributeInTypelist<KBoundaryTypes>(hashGenerator, optarg);
                OmitAttributeInTypelist<KShapeTypes>(hashGenerator, optarg);
                break;
            default:  // unrecognized option
                std::cout << usage << std::endl;
                return 1;
        }
    }

    std::string inFileName = argv[optind];
    std::string fileSuffix = inFileName.substr(inFileName.find_last_of('.'), std::string::npos);

    struct stat fileInfo;
    bool exists;
    int fileStat;

    // Attempt to get the file attributes
    fileStat = stat(inFileName.c_str(), &fileInfo);
    if (fileStat == 0)
        exists = true;
    else
        exists = false;

    if (!exists) {
        std::cout << "Error: file \"" << inFileName << "\" cannot be read." << std::endl;
        return 1;
    }

    KBinaryDataStreamer binaryDataStreamer;

    if (fileSuffix != binaryDataStreamer.GetFileSuffix()) {
        std::cout << "Error: unkown file extension \"" << suffix << "\"" << std::endl;
        return 1;
    }

    KEMFileInterface::GetInstance()->Read(inFileName, surfaceContainer, name);

    std::cout << hashGenerator.GenerateHash(surfaceContainer) << std::endl;

    return 0;
}
