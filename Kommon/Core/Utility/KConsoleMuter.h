/**
 * @file KConsoleMuter.h
 *
 * @date 24.12.2013
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#ifndef KCONSOLEMUTER_H_
#define KCONSOLEMUTER_H_

#include "KNonCopyable.h"

#include <cstdio>
#include <fstream>
#include <vector>

namespace katrin
{

class KConsoleMuter : KNonCopyable
{
  public:
    KConsoleMuter() = default;
    KConsoleMuter(std::ostream& stream);
    KConsoleMuter(FILE* stream);
    virtual ~KConsoleMuter();

    void Mute(std::ostream& stream);
    void Mute(FILE* stream);
    void UnMute();

  protected:
    std::ofstream fNullStream;
    std::vector<std::ostream*> fMutedStreams;
    std::vector<std::streambuf*> fBackupStreamBufs;
    std::vector<FILE*> fMutedCStreams;
};

inline KConsoleMuter::KConsoleMuter(std::ostream& stream)
{
    Mute(stream);
}

inline KConsoleMuter::KConsoleMuter(FILE* cStream)
{
    Mute(cStream);
}

inline KConsoleMuter::~KConsoleMuter()
{
    UnMute();
}

inline void KConsoleMuter::Mute(std::ostream& stream)
{
    if (!fNullStream.is_open())
        fNullStream.open("/dev/null", std::ofstream::app);

    fMutedStreams.push_back(&stream);
    fBackupStreamBufs.push_back(stream.rdbuf());
    stream.rdbuf(fNullStream.rdbuf());
}

inline void KConsoleMuter::Mute(FILE* cStream)
{
    if (!cStream)
        return;

    fMutedCStreams.push_back(cStream);
    freopen("/dev/null", "a", cStream);
}

inline void KConsoleMuter::UnMute()
{
    for (size_t i = 0; i < fMutedStreams.size(); ++i) {
        fMutedStreams[i]->rdbuf(fBackupStreamBufs[i]);
    }
    fMutedStreams.clear();
    fBackupStreamBufs.clear();

    // this might not restore the C streams correctly:
    for (auto& stream : fMutedCStreams) {
        freopen("/dev/tty", "a", stream);
    }
    fMutedCStreams.clear();

    fNullStream.close();
}

} /* namespace katrin */

#endif /* KCONSOLEMUTER_H_ */
