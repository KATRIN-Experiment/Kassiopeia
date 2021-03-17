#ifndef Kassiopeia_KSCommandGroup_h_
#define Kassiopeia_KSCommandGroup_h_

#include "KSCommand.h"

namespace Kassiopeia
{

class KSCommandGroup : public KSCommand
{
  public:
    KSCommandGroup();
    KSCommandGroup(const KSCommandGroup& aCopy);
    ~KSCommandGroup() override;

  public:
    KSCommandGroup* Clone() const override;

  public:
    void AddCommand(KSCommand* anCommand);
    void RemoveCommand(KSCommand* anCommand);

    KSCommand* CommandAt(unsigned int anIndex);
    const KSCommand* CommandAt(unsigned int anIndex) const;
    unsigned int CommandCount() const;

  private:
    typedef std::vector<KSCommand*> CommandVector;
    using CommandIt = CommandVector::iterator;
    using CommandCIt = CommandVector::const_iterator;

    CommandVector fCommands;
};

}  // namespace Kassiopeia

#endif
