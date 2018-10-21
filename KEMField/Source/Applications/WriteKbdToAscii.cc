// WriteKbdToAscii
// This program writes information on the electrode shape and charge density into a single text file.
// Author: Daniel Hilk
// Date: 24.03.2016

#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <sys/stat.h>

#include "KTypelist.hh"
#include "KSurfaceContainer.hh"
#include "KEMFileInterface.hh"
#include "KBinaryDataStreamer.hh"

#include "KSADataStreamer.hh"
#include "KSerializer.hh"
#include "KSADataStreamer.hh"

using namespace KEMField;

namespace KEMField {

struct electrodeData {
	unsigned short type; // 0 = triangle, 1 = rectangle, 2 = line segment
	double shape[11];
	double cd; // charge density
};

class BasisDataExtractor : public KSelectiveVisitor<KBasisVisitor,KTYPELIST_1(KElectrostaticBasis)>
{
public:
	BasisDataExtractor(){};
	virtual ~BasisDataExtractor(){};

	using KSelectiveVisitor<KBasisVisitor, KTYPELIST_1(KElectrostaticBasis)>::Visit;

	void Visit(KElectrostaticBasis& basis)
	{
		fCurrentBasisData = basis.GetSolution(0);
	}

	double GetBasisData() const {return fCurrentBasisData;};

private:

	double fCurrentBasisData;
};

class ShapeDataExtractor : public KSelectiveVisitor<KShapeVisitor, KTYPELIST_3(KTriangle,KRectangle,KLineSegment)>
{
public:
	using KSelectiveVisitor<KShapeVisitor,KTYPELIST_3(KTriangle,KRectangle,KLineSegment)>::Visit;

	ShapeDataExtractor(){};
	virtual ~ShapeDataExtractor(){};

	void Visit(KTriangle& t) { ProcessTriangle(t); }
	void Visit(KRectangle& r) { ProcessRectangle(r); }
	void Visit(KLineSegment& l) { ProcessLineSegment(l); }

	void ProcessTriangle(KTriangle& t)
	{
		currentElectrode.type = 0;
		currentElectrode.shape[0] = t.GetA();
		currentElectrode.shape[1] = t.GetB();
		currentElectrode.shape[2] = t.GetP0().X();
		currentElectrode.shape[3] = t.GetP0().Y();
		currentElectrode.shape[4] = t.GetP0().Z();
		currentElectrode.shape[5] = t.GetN1().X();
		currentElectrode.shape[6] = t.GetN1().Y();
		currentElectrode.shape[7] = t.GetN1().Z();
		currentElectrode.shape[8] = t.GetN2().X();
		currentElectrode.shape[9] = t.GetN2().Y();
		currentElectrode.shape[10] = t.GetN2().Z();
		currentElectrode.cd = 0.;

		return;
	}

	void ProcessRectangle(KRectangle& r)
	{
		currentElectrode.type = 1;
		currentElectrode.shape[0] = r.GetA();
		currentElectrode.shape[1] = r.GetB();
		currentElectrode.shape[2] = r.GetP0().X();
		currentElectrode.shape[3] = r.GetP0().Y();
		currentElectrode.shape[4] = r.GetP0().Z();
		currentElectrode.shape[5] = r.GetN1().X();
		currentElectrode.shape[6] = r.GetN1().Y();
		currentElectrode.shape[7] = r.GetN1().Z();
		currentElectrode.shape[8] = r.GetN2().X();
		currentElectrode.shape[9] = r.GetN2().Y();
		currentElectrode.shape[10] = r.GetN2().Z();
		currentElectrode.cd = 0.;

		return;
	}

	void ProcessLineSegment(KLineSegment& l)
	{
		currentElectrode.type = 2;
		currentElectrode.shape[0] = l.GetP0().X();
		currentElectrode.shape[1] = l.GetP0().Y();
		currentElectrode.shape[2] = l.GetP0().Z();
		currentElectrode.shape[3] = l.GetP1().X();
		currentElectrode.shape[4] = l.GetP1().Y();
		currentElectrode.shape[5] = l.GetP1().Z();
		currentElectrode.shape[6] = l.GetDiameter();
		for( unsigned short i=7; i<11; i++) {
			currentElectrode.shape[i] = 0.;
		}
		currentElectrode.cd = 0.;
		return;
 	}

	electrodeData GetShapeData() const { return currentElectrode; }

private:
	electrodeData currentElectrode;
};

} /* KEMField namespace*/

void WritePlaceholder( std::ostream &tri, std::ostream &rect, std::ostream &line )
{
	tri << "placeholder" << "\n";
	rect << "placeholder" << "\n";
	line << "placeholder" << "\n";
}

void WritePlainText( std::ostream &tri, std::ostream &rect, std::ostream &line, electrodeData input )
{
	if( input.type==0 ) {
		tri << std::scientific << std::setprecision(16);
		for( unsigned short i=0; i<11; i++ ) {
			tri << input.shape[i] << "\t";
		}
		tri << input.cd << "\n";
	}
	if( input.type==1 ) {
		rect << std::scientific << std::setprecision(16);
		for( unsigned short i=0; i<11; i++ ) {
			rect << input.shape[i] << "\t";
		}
		rect << input.cd << "\n";
	}
	if( input.type==2 ) {
		line << std::scientific << std::setprecision(16);
		for( unsigned short i=0; i<11; i++ ) {
			line << input.shape[i] << "\t";
		}
		line << input.cd << "\n";
	}
}


int main(int argc, char* argv[])
{

    std::string usage =
    "\n"
    "Usage: WriteKbdToAscii <options>\n"
    "\n"
    "This program translates .kbd files into three ASCII files.\n"
    "\n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\t -f, --file               (specify the input kbd file)\n"
    "\n";

    static struct option longOptions[] = {
        {"help", no_argument, 0, 'h'},
        {"file", required_argument, 0, 'f'},
        {"name", required_argument, 0, 'n'}
    };

    static const char *optString = "ha:b:n:m:s:";

    std::string inFile = "";
    std::string containerName = "surfaceContainer";

    while(1)
    {
        char optId = getopt_long(argc, argv,optString, longOptions, NULL);
        if(optId == -1) break;
        switch(optId) {
        case('h'): // help
            std::cout<<usage<<std::endl;
        break;
        case('f'):
            inFile = std::string(optarg);
        break;
        case ('n'):
            containerName = std::string(optarg);
        break;
        default: // unrecognized option
            std::cout<<usage<<std::endl;
        return 1;
        }
    }

    std::string suffix = inFile.substr(inFile.find_last_of("."),std::string::npos);

    struct stat fileInfo;
    bool exists;
    int fileStat;

    // Attempt to get the file attributes
    fileStat = stat(inFile.c_str(),&fileInfo);
    if(fileStat == 0)
    exists = true;
    else
    exists = false;

    if (!exists) {
		std::cout << "Error: file \"" << inFile <<"\" cannot be read." << std::endl;
		return 1;
    }

    KBinaryDataStreamer binaryDataStreamer;

    if (suffix.compare(binaryDataStreamer.GetFileSuffix()) != 0) {
        std::cout<<"Error: unkown file extension \""<<suffix<<"\"" << std::endl;
        return 1;
    }

    //inspect the files
    KEMFileInterface::GetInstance()->Inspect(inFile);

    //now read in the surface containers
    KSurfaceContainer container;
    KEMFileInterface::GetInstance()->Read(inFile,container,containerName);

    std::cout << "Surface container with name " << containerName << " in file has size: " << container.size() << std::endl;


    unsigned int size = container.size();
    //loop over every element in the container and retrieve shape data and the charge density

    electrodeData electrodeIt;
    std::vector<electrodeData> data;

    ShapeDataExtractor visShape;
    BasisDataExtractor visBasis;

    for( unsigned int i=0; i < size; i++ ) {
    	//extract the shape and basis data
    	container.at(i)->Accept(visShape);
    	container.at(i)->Accept(visBasis);

    	electrodeIt = visShape.GetShapeData();
    	electrodeIt.cd = visBasis.GetBasisData();

    	data.push_back( electrodeIt );
    }

    std::ofstream triFile;
    triFile.open("triangles.txt");
    std::ofstream rectFile;
    rectFile.open("rectangles.txt");
    std::ofstream wireFile;
    wireFile.open("wires.txt");

    // write first line as placeholder
	WritePlaceholder( triFile, rectFile, wireFile );

    for( unsigned int i=0; i<data.size(); i++ ) {
    	WritePlainText( triFile, rectFile, wireFile, data.at(i) );
    }

    triFile.close();
    rectFile.close();
    wireFile.close();

  return 0;
}
