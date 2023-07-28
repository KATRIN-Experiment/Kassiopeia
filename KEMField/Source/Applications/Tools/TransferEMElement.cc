#include "KBinaryDataStreamer.hh"
#include "KDataDisplay.hh"
#include "KEMConstants.hh"
#include "KEMCout.hh"
#include "KEMFileInterface.hh"
#include "KIterativeStateWriter.hh"
#include "KResidualVector.hh"
#include "KSADataStreamer.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KTypelist.hh"

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

using namespace KEMField;

template<class Element>
void TransferElement(std::string& inFile, std::string& inName, std::string& outFile, std::string& outName,
                     std::vector<std::string>& labels)
{
    Element element;
    std::cout << "IN: \t" << inFile << std::endl;
    std::cout << "\t" << inName << std::endl;
    std::cout << "\t" << Element::Name() << std::endl;
    KEMFileInterface::GetInstance()->Read(inFile, element, inName);
    std::cout << "OUT: \t" << outFile << std::endl;
    std::cout << "\t" << outName << std::endl;
    for (auto& label : labels)
        std::cout << "\t" << label << std::endl;
    KEMFileInterface::GetInstance()->Write(outFile, element, outName, labels);
}

template<class Typelist> struct TransferElementInTypelist
{
    TransferElementInTypelist(const std::string& elementName, std::string& inFile, std::string& inName,
                              std::string& outFile, std::string& outName, std::vector<std::string>& labels)
    {
        typedef typename Typelist::Head Head;
        using Tail = typename Typelist::Tail;

        if (Head::Name() == elementName)
            TransferElement<Head>(inFile, inName, outFile, outName, labels);
        else
            TransferElementInTypelist<Tail>(elementName, inFile, inName, outFile, outName, labels);
    }
};

template<> struct TransferElementInTypelist<KNullType>
{
    TransferElementInTypelist(const std::string& elementName, std::string& /*unused*/, std::string& /*unused*/,
                              std::string& /*unused*/, std::string& /*unused*/, std::vector<std::string>& /*unused*/)
    {
        KEMField::cout << "Unknown type <" << elementName << ">" << KEMField::endl;
    }
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
    AppendNames(const std::string& /*unused*/, const std::string /*unused*/&, std::string& /*unused*/) {}
};

int main(int argc, char* argv[])
{
    using KElementTypes = KEMField::KTypelist<
        KSurfaceContainer,
        KEMField::KTypelist<KResidualVector<double>, KEMField::KTypelist<KResidualThreshold, KEMField::KNullType>>>;

    std::string usage = "\n"
                        "Usage: TransferEMElement -i <InElementFile> -t <ElementType>\n"
                        "\n"
                        "This program takes a streamed KEMField element and saves it to a new file.\n"
                        "\n"
                        "\tAvailable options:\n"
                        "\t -h, --help               (shows this message and exits)\n"
                        "\t -i, --in_file            (name of file from which element is read)\n"
                        "\t -o, --out_file           (name of file to which element is written)\n"
                        "\t -t, --type               (type of element that is read)\n"
                        "\t -n, --in_name            (name of the element that is read)\n"
                        "\t -m, --out_name           (name of the element that is written)\n"
                        "\t -l, --label              (adds a label to the stored element)\n"
                        "\n"
                        "\tElement types:\n";

    std::string prefix = "\t\t";
    std::string suffix = ",\n";

    AppendNames<KElementTypes>(prefix, suffix, usage);
    usage = usage.substr(0, usage.find_last_of(suffix) - (suffix.size() - 1));
    usage.append("\n\n");


    if (argc == 1) {
        std::cout << usage;
        return 0;
    }

    static struct option longOptions[] = {
        {"help", no_argument, nullptr, 'h'},
        {"in_file", required_argument, nullptr, 'i'},
        {"out_file", required_argument, nullptr, 'o'},
        {"type", required_argument, nullptr, 't'},
        {"in_name", required_argument, nullptr, 'n'},
        {"out_name", required_argument, nullptr, 'm'},
        {"label", required_argument, nullptr, 'l'},
    };

    static const char* optString = "hi:o:t:n:m:l:";

    std::string inFile;
    std::string outFile;
    std::string type;
    std::string inName;
    std::string outName;
    std::vector<std::string> labels;

    while (true) {
        int optId = getopt_long(argc, argv, optString, longOptions, nullptr);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                return 0;
            case ('i'):  // in_file
                inFile = optarg;
                break;
            case ('o'):  // out_file
                outFile = optarg;
                break;
            case ('t'):  // type
                type = optarg;
                break;
            case ('n'):  // in_name
                inName = optarg;
                break;
            case ('m'):  // out_name
                outName = optarg;
                break;
            case ('l'):  // label
                labels.emplace_back(optarg);
                break;
            default:  // unrecognized option
                std::cout << usage << std::endl;
                return 1;
        }
    }

    if (inFile.empty() || inName.empty()) {
        std::cout << usage << std::endl;
        return 1;
    }

    if (outFile.empty())
        outFile = KEMFileInterface::GetInstance()->GetActiveFileName();

    if (outName.empty())
        outName = inName;

    std::string fileSuffix = inFile.substr(inFile.find_last_of('.'), std::string::npos);

    struct stat fileInfo;
    bool exists;
    int fileStat;

    // Attempt to get the file attributes
    fileStat = stat(inFile.c_str(), &fileInfo);
    if (fileStat == 0)
        exists = true;
    else
        exists = false;

    if (!exists) {
        std::cout << "Error: file \"" << inFile << "\" cannot be read." << std::endl;
        return 1;
    }

    KBinaryDataStreamer binaryDataStreamer;

    if (fileSuffix != binaryDataStreamer.GetFileSuffix()) {
        std::cout << "Error: unkown file extension \"" << suffix << "\"" << std::endl;
        return 1;
    }

    TransferElementInTypelist<KElementTypes>(type, inFile, inName, outFile, outName, labels);

    return 0;
}
