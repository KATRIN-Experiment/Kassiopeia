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
    std::string SignalName() const override
    {
        return "field_error";
    };
};

class KSGeneratorError : public katrin::KExceptionPrototype<KSGeneratorError, KSException>
{
  public:
    std::string SignalName() const override
    {
        return "generator_error";
    };
};

class KSNavigatorError : public katrin::KExceptionPrototype<KSNavigatorError, KSException>
{
  public:
    std::string SignalName() const override
    {
        return "navigator_error";
    };
};

class KSInteractionError : public katrin::KExceptionPrototype<KSInteractionError, KSException>
{
  public:
    std::string SignalName() const override
    {
        return "interaction_error";
    };
};

class KSTrajectoryError : public katrin::KExceptionPrototype<KSTrajectoryError, KSException>
{
  public:
    std::string SignalName() const override
    {
        return "trajectory_error";
    };
};

class KSTerminatorError : public katrin::KExceptionPrototype<KSTerminatorError, KSException>
{
  public:
    std::string SignalName() const override
    {
        return "terminatior_error";
    };
};

class KSWriterError : public katrin::KExceptionPrototype<KSWriterError, KSException>
{
  public:
    std::string SignalName() const override
    {
        return "writer_error";
    };
};

class KSModifierError : public katrin::KExceptionPrototype<KSModifierError, KSException>
{
  public:
    std::string SignalName() const override
    {
        return "modifier_error";
    };
};

}  // namespace Kassiopeia

#endif
