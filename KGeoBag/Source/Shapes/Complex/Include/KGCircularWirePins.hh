#ifndef KGCIRCULARWIREPINS_DEF
#define KGCIRCULARWIREPINS_DEF

#include <stddef.h>
#include <vector>
#include <cmath>
#include <string>

#include "KGBoundary.hh"

namespace KGeoBag
{

class KGCircularWirePins : public KGBoundary
{
	/*
	 A class describing circular wire pins with a flat section profile
	 */
public:
	KGCircularWirePins() {
	}
	KGCircularWirePins(double r1,
			double r2,
			unsigned int nPins,
			double diameter,
			double rotationAngle,
			unsigned int nDisc,
			unsigned int nDiscPower) :	fR1(r1),
					fR2(r2),
					fNPins(nPins),
					fDiameter(diameter),
					fRotationAngle(rotationAngle),
					fNDisc(nDisc),
					fNDiscPower(nDiscPower)
	{
	}

	virtual ~KGCircularWirePins() {
	}

	static std::string Name() {
		return "circular_wire_pins";
	}

	virtual KGCircularWirePins* Clone() const;

	virtual void Initialize() const {
	}

	bool ContainsPoint(const double* P) const;
	double DistanceTo(const double* P, double* P_in = NULL, double* P_norm =
			NULL) const;

	double GetLength() const;
	double Area() const;
	double Volume() const;

	void SetR1(double d) {
		fR1 = d;
	}
	void SetR2(double d) {
		fR2 = d;
	}
	void SetNPins(unsigned int d) {
		fNPins = d;
	}
	void SetDiameter(double d) {
		fDiameter = d;
	}
	void SetRotationAngle(double d) {
		fRotationAngle = d;
	}
	void SetNDisc(unsigned int d) {
		fNDisc = d;
	}
	void SetNDiscPower(double d) {
		fNDiscPower = d;
	}

	double GetR1() const {
		return fR1;
	}
	double GetR2() const {
		return fR2;
	}
	unsigned int GetNPins() const {
		return fNPins;
	}
	double GetDiameter() const {
		return fDiameter;
	}
	double GetRotationAngle() const {
			return fRotationAngle;
		}
	unsigned int GetNDisc() const {
		return fNDisc;
	}
	double GetNDiscPower() const {
		return fNDiscPower;
	}

private:

	double fR1;
	double fR2;
	unsigned int fNPins;
	double fDiameter;
	double fRotationAngle;
	unsigned int fNDisc;
	double fNDiscPower;
};
}

#endif /* KGCIRCULARWIREPINS_DEF */
