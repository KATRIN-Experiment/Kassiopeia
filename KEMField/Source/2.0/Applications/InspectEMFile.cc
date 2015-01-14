#include <getopt.h>
#include <iostream>
#include <sys/stat.h>

#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"

using namespace KEMField;

int main(int argc, char* argv[])
{
  std::string usage =
    "\n"
    "Usage: InspectEMFile <File.kbd>\n"
    "\n"
    "This program takes a KEMField file and prints its keys.\n"
    "\n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\n";

  if (argc == 1)
  {
    std::cout<<usage;
    return 0;
  }

  static struct option longOptions[] = {
    {"help", no_argument, 0, 'h'},
  };

  static const char *optString = "h";

  while(1) {
    char optId = getopt_long(argc, argv,optString, longOptions, NULL);
    if(optId == -1) break;
    switch(optId) {
    case('h'): // help
      //
    default: // unrecognized option
      std::cout<<usage<<std::endl;
      return 1;
    }
  }

  std::string inFileName = argv[optind];

  std::string suffix = inFileName.substr(inFileName.find_last_of("."),std::string::npos);

  struct stat fileInfo;
  bool exists;
  int fileStat;

  // Attempt to get the file attributes
  fileStat = stat(inFileName.c_str(),&fileInfo);
  if(fileStat == 0)
    exists = true;
  else
    exists = false;

  if (!exists)
  {
    std::cout<<"Error: file \""<<inFileName<<"\" cannot be read."<<std::endl;
    return 1;
  }

  KBinaryDataStreamer binaryDataStreamer;

  if (suffix.compare(binaryDataStreamer.GetFileSuffix()) != 0)
  {
    std::cout<<"Error: unkown file extension \""<<suffix<<"\""<<std::endl;
    return 1;
  }

  KEMFileInterface::GetInstance()->Inspect(inFileName);

  return 0;
}
