/**
 * @file
 * @author Jan Behrens <jan.behrens@kit.edu>
 */

#ifndef Kassiopeia_KSException_h_
#define Kassiopeia_KSException_h_

// Kasper includes
#include "KException.h"

namespace Kassiopeia
{

class KSUserInterrupt : public katrin::KExceptionPrototype<KSUserInterrupt, katrin::KException>
{};


class KSException : public katrin::KExceptionPrototype<KSException, katrin::KException>
{
  public:
    virtual std::string SignalName() const = 0;
};

class KSFieldError : public katrin::KExceptionPrototype<KSFieldError, KSException>
{
  public:
    virtual std::string SignalName() const
    {
        return "field_error";
    };
};

class KSGeneratorError : public katrin::KExceptionPrototype<KSGeneratorError, KSException>
{
  public:
    virtual std::string SignalName() const
    {
        return "generator_error";
    };
};

class KSNavigatorError : public katrin::KExceptionPrototype<KSNavigatorError, KSException>
{
  public:
    virtual std::string SignalName() const
    {
        return "navigator_error";
    };
};

class KSInteractionError : public katrin::KExceptionPrototype<KSInteractionError, KSException>
{
  public:
    virtual std::string SignalName() const
    {
        return "interaction_error";
    };
};

class KSTrajectoryError : public katrin::KExceptionPrototype<KSTrajectoryError, KSException>
{
  public:
    virtual std::string SignalName() const
    {
        return "trajectory_error";
    };
};

class KSTerminatorError : public katrin::KExceptionPrototype<KSTerminatorError, KSException>
{
  public:
    virtual std::string SignalName() const
    {
        return "terminatior_error";
    };
};

class KSWriterError : public katrin::KExceptionPrototype<KSWriterError, KSException>
{
  public:
    virtual std::string SignalName() const
    {
        return "writer_error";
    };
};

class KSModifierError : public katrin::KExceptionPrototype<KSModifierError, KSException>
{
  public:
    virtual std::string SignalName() const
    {
        return "modifier_error";
    };
};

}  // namespace Kassiopeia

#endif
