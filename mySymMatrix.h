#pragma once

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#include "myQuaternion.h"

struct symMatrix
{
	double xx{ 0. };
	double yy{ 0. };
	double zz{ 0. };
	double xy{ 0. };
	double xz{ 0. };
	double yz{ 0. };
};

HOST_DEVICE inline symMatrix make_symMatrix(double xx, double yy, double zz, double xy, double xz, double yz)
{
	symMatrix m;
	m.xx = xx;
	m.yy = yy;
	m.zz = zz;
	m.xy = xy;
	m.xz = xz;
	m.yz = yz;
	return m;
}

HOST_DEVICE inline double norm(const symMatrix& m)
{
	return sqrt(2 * (m.xx * m.xx + m.yy * m.yy + m.zz * m.zz + 2 * (m.xy * m.xy + m.xz * m.xz + m.yz * m.yz)));
}

HOST_DEVICE inline symMatrix deviatoric(const symMatrix& m)
{
	double tr = (m.xx + m.yy + m.zz) / 3.;
	return make_symMatrix(m.xx - tr, m.yy - tr, m.zz - tr, m.xy, m.xz, m.yz);
}

HOST_DEVICE inline symMatrix operator+(const symMatrix& m1, const symMatrix& m2)
{
	symMatrix mm;
	mm.xx = m1.xx + m2.xx;
	mm.yy = m1.yy + m2.yy;
	mm.zz = m1.zz + m2.zz;
	mm.xy = m1.xy + m2.xy;
	mm.xz = m1.xz + m2.xz;
	mm.yz = m1.yz + m2.yz;
	return mm;
}

HOST_DEVICE inline symMatrix operator-(const symMatrix& m1, const symMatrix& m2)
{
	symMatrix mm;
	mm.xx = m1.xx - m2.xx;
	mm.yy = m1.yy - m2.yy;
	mm.zz = m1.zz - m2.zz;
	mm.xy = m1.xy - m2.xy;
	mm.xz = m1.xz - m2.xz;
	mm.yz = m1.yz - m2.yz;
	return mm;
}

HOST_DEVICE inline symMatrix operator*(const symMatrix& m, double a)
{
	symMatrix mm;
	mm.xx = m.xx * a;
	mm.yy = m.yy * a;
	mm.zz = m.zz * a;
	mm.xy = m.xy * a;
	mm.xz = m.xz * a;
	mm.yz = m.yz * a;
	return mm;
}

HOST_DEVICE inline symMatrix operator*(double a, const symMatrix& m)
{
	symMatrix mm;
	mm.xx = m.xx * a;
	mm.yy = m.yy * a;
	mm.zz = m.zz * a;
	mm.xy = m.xy * a;
	mm.xz = m.xz * a;
	mm.yz = m.yz * a;
	return mm;
}

HOST_DEVICE inline double3 operator*(const symMatrix& m, const double3& v)
{
	return make_double3(m.xx * v.x + m.xy * v.y + m.xz * v.z, m.xy * v.x + m.yy * v.y + m.yz * v.z, m.xz * v.x + m.yz * v.y + m.zz * v.z);
}

HOST_DEVICE inline double3 operator*(const double3& v, const symMatrix& m) 
{
	return make_double3(m.xx * v.x + m.xy * v.y + m.xz * v.z, m.xy * v.x + m.yy * v.y + m.yz * v.z, m.xz * v.x + m.yz * v.y + m.zz * v.z);
};

HOST_DEVICE inline symMatrix rotateInverseInertiaTensor(const quaternion& q, const symMatrix& invI) 
{
	double q0 = q.q0, q1 = q.q1, q2 = q.q2, q3 = q.q3;
	double a = 1 - 2 * q2 * q2 - 2 * q3 * q3;
	double b = 2 * q1 * q2 - 2 * q0 * q3;
	double c = 2 * q1 * q3 + 2 * q0 * q2;
	double d = 2 * q1 * q2 + 2 * q0 * q3;
	double e = 1 - 2 * q1 * q1 - 2 * q3 * q3;
	double f = 2 * q2 * q3 - 2 * q0 * q1;
	double g = 2 * q1 * q3 - 2 * q0 * q2;
	double h = 2 * q2 * q3 + 2 * q0 * q1;
	double i = 1 - 2 * q1 * q1 - 2 * q2 * q2;
	double m_xx = invI.xx * a * a + 2 * invI.xy * a * b + 2 * invI.xz * a * c + invI.yy * b * b + 2 * invI.yz * b * c +
		invI.zz * c * c,
		m_xy = d * (invI.xx * a + invI.xy * b + invI.xz * c) + e * (invI.xy * a + invI.yy * b + invI.yz * c) +
		f * (invI.xz * a + invI.yz * b + invI.zz * c),
		m_xz = g * (invI.xx * a + invI.xy * b + invI.xz * c) + h * (invI.xy * a + invI.yy * b + invI.yz * c) +
		i * (invI.xz * a + invI.yz * b + invI.zz * c),
		m_yy = invI.xx * d * d + 2 * invI.xy * d * e + 2 * invI.xz * d * f + invI.yy * e * e + 2 * invI.yz * e * f +
		invI.zz * f * f,
		m_yz = g * (invI.xx * d + invI.xy * e + invI.xz * f) + h * (invI.xy * d + invI.yy * e + invI.yz * f) +
		i * (invI.xz * d + invI.yz * e + invI.zz * f),
		m_zz = invI.xx * g * g + 2 * invI.xy * g * h + 2 * invI.xz * g * i + invI.yy * h * h + 2 * invI.yz * h * i +
		invI.zz * i * i;
	return make_symMatrix(m_xx, m_yy, m_zz, m_xy, m_xz, m_yz);
};

HOST_DEVICE inline symMatrix inverse(const symMatrix& A)
{
	double det = 1. / (A.xx * (A.yy * A.zz - A.yz * A.yz)
		+ A.xy * (A.yz * A.xz - A.xy * A.zz)
		+ A.xz * (A.xy * A.yz - A.yy * A.xz));
	symMatrix B;
	B.xx = (A.yy * A.zz - A.yz * A.yz) * det;
	B.xy = -(A.xy * A.zz - A.xz * A.yz) * det;
	B.xz = (A.xy * A.yz - A.xz * A.yy) * det;
	B.yy = (A.xx * A.zz - A.xz * A.xz) * det;
	B.yz = -(A.xx * A.yz - A.xz * A.xy) * det;
	B.zz = (A.xx * A.yy - A.xy * A.xy) * det;
	return B;
};