#include "KBinaryDataStreamer.hh"

#include "KEMCoreMessage.hh"

#include <algorithm>
#include <iostream>
#include <locale>

namespace KEMField
{

void KBinaryDataStreamer::open(const std::string& fileName, const std::string& action)
{
    std::string action_;
    action_.resize(action.length());

    std::transform(action.begin(), action.end(), action_.begin(), ::toupper);

    if (action_ == "READ") {
        fFile.open(fileName.c_str(), std::fstream::in | std::ios::binary);
    }
    if (action_ == "MODIFY") {
        fFile.open(fileName.c_str(), std::fstream::in | std::fstream::out | std::ios::binary);
    }
    if (action_ == "UPDATE") {
        fFile.open(fileName.c_str(), std::fstream::out | std::fstream::app | std::ios::binary);
    }
    if (action_ == "OVERWRITE")
        fFile.open(fileName.c_str(), std::fstream::out | std::ios::binary);

    if (!fFile.is_open() || !fFile.good()) {
        // NOTE: dependent code uses this method also for non-existent files, so don't show an error here.
        kem_cout((action_ == "MODIFY" || action_ == "UPDATE") ? eWarning : eInfo)
                << "Cannot open file <" << fileName << "> to " << action_ << "." << eom;
        return;
    }
}

void KBinaryDataStreamer::close()
{
    if (fFile.is_open())
        fFile.close();
}

}  // namespace KEMField
