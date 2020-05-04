// WriteAsciiToMSH
// This program converts the ASCII output triangles.txt of WriteKbdToAscii into an MSH file.
// Author: Zachary Bogorad
// Date: 07.02.2016

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <typeinfo>
#include <vector>

int main(int argc, char* argv[])
{

    std::string usage = "\n"
                        "Usage: WriteAsciiToPCD <options>\n"
                        "\n"
                        "This program translates ASCII triangles.txt files from WriteKbdToAscii into PCD files.\n"
                        "\n"
                        "\tAvailable options:\n"
                        "\t -h, --help               (shows this message and exits)\n"
                        "\t -f, --file               (specify the input triangles.txt file)\n"
                        "\n";

    static struct option longOptions[] = {{"help", no_argument, nullptr, 'h'},
                                          {"file", required_argument, nullptr, 'f'}};

    static const char* optString = "ha:b:n:m:s:";

    std::string inFile = "";

    while (true) {
        char optId = getopt_long(argc, argv, optString, longOptions, nullptr);
        if (optId == -1) {
            break;
        }
        switch (optId) {
            case ('h'):  // help
                std::cout << usage << std::endl;
                break;
            case ('f'):
                inFile = std::string(optarg);
                break;
            default:  // unrecognized option
                return 1;
        }
    }

    std::string suffix = inFile.substr(inFile.find_last_of("."), std::string::npos);

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

    if (suffix.compare(".txt") != 0) {
        std::cout << "Error: unkown file extension \"" << suffix << "\"" << std::endl;
        return 1;
    }

    std::string line;
    std::ifstream fileReader(inFile);
    std::ofstream fileWriter("mesh.pcd");

    std::vector<double> xlist;
    std::vector<double> ylist;
    std::vector<double> zlist;
    std::vector<double> xnlist;
    std::vector<double> ynlist;
    std::vector<double> znlist;

    while (getline(fileReader, line)) {
        //Input format is L1 | L2 | P0.x | P0.y | P0.z | N1.x | N1.y | N1.z | N2.x | N2.y | N2.z | (charge density value)
        std::string temp = line;
        char* token = strtok(&temp[0], "\t");
        if (token != nullptr && token[0] != 'p') {
            //Skip lengths
            token = strtok(nullptr, "\t");
            token = strtok(nullptr, "\t");

            // Get coordinates
            xlist.push_back(std::stod(token));
            token = strtok(nullptr, "\t");
            ylist.push_back(std::stod(token));
            token = strtok(nullptr, "\t");
            zlist.push_back(std::stod(token));
            token = strtok(nullptr, "\t");

            //Get edge vectors
            double xn1 = std::stod(token);
            token = strtok(nullptr, "\t");
            double yn1 = std::stod(token);
            token = strtok(nullptr, "\t");
            double zn1 = std::stod(token);
            token = strtok(nullptr, "\t");

            double xn2 = std::stod(token);
            token = strtok(nullptr, "\t");
            double yn2 = std::stod(token);
            token = strtok(nullptr, "\t");
            double zn2 = std::stod(token);
            token = strtok(nullptr, "\t");

            // Calculate normal vector
            double xn = yn1 * zn2 - yn2 * zn1;
            xnlist.push_back(xn);
            double yn = zn1 * xn2 - zn2 * xn1;
            ynlist.push_back(yn);
            double zn = xn1 * yn2 - xn2 * yn1;
            znlist.push_back(zn);

            //Skip charge
            token = strtok(nullptr, "\t");
        }
        // while (token != NULL)
        // {
        //   //std::cout << "Token: " << token << "\n";
        //   if(token!=NULL && token[0]!='p'){
        //     double value = std::stod(token);
        //     //std::cout << "Value: " << value << "\n";
        //   }
        //   else{
        //     //std::cout << "Not a number.\n";
        //   }
        //   token = strtok (NULL, "\t");
        // }
    }
    fileReader.close();

    // Write the header

    fileWriter << "VERSION .7\n";
    //fileWriter << "FIELDS x y z normal_x normal_y normal_z\n";
    fileWriter << "FIELDS x y z\n";
    // fileWriter << "SIZE 8 8 8 8 8 8\n";
    fileWriter << "SIZE 8 8 8\n";
    // fileWriter << "TYPE F F F F F F\n";
    fileWriter << "TYPE F F F\n";
    // fileWriter << "COUNT 1 1 1 1 1 1\n";
    fileWriter << "COUNT 1 1 1\n";
    fileWriter << "WIDTH " << xlist.size() << "\n";
    fileWriter << "HEIGHT 1\n";
    //fileWriter << "VIEWPOINT 0 0 0 1 0 0 0\n";  //optional
    fileWriter << "POINTS " << xlist.size() << "\n";
    fileWriter << "DATA ascii\n";

    for (unsigned int i = 0; i < xlist.size(); i++) {
        // fileWriter << xlist[i] << " " << ylist[i] << " " << zlist[i] << " " << xnlist[i] << " " << ynlist[i] << " " << znlist[i] << "\n";
        fileWriter << xlist[i] << " " << ylist[i] << " " << zlist[i] << "\n";
    }

    fileWriter.close();

    return 0;
}
