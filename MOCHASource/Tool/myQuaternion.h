#pragma once

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#include "myVector.h"

struct quaternion {
	double q0;
	double q1;
	double q2;
	double q3;
};

HOST_DEVICE inline quaternion make_quaternion(double q0, double q1, double q2, double q3) {
	quaternion q;
	q.q0 = q0, q.q1 = q1, q.q2 = q2, q.q3 = q3;
	return q;
};

HOST_DEVICE inline quaternion operator+(const quaternion& a, const quaternion& b) {
	return make_quaternion(a.q0 + b.q0, a.q1 + b.q1, a.q2 + b.q2, a.q3 + b.q3);
};

HOST_DEVICE inline quaternion operator*(const quaternion& q, double c) {
	return make_quaternion(q.q0 * c, q.q1 * c, q.q2 * c, q.q3 * c);
};

HOST_DEVICE inline quaternion operator*(double c, const quaternion& q) {
	return make_quaternion(q.q0 * c, q.q1 * c, q.q2 * c, q.q3 * c);
};

HOST_DEVICE inline quaternion operator/(const quaternion& q, double c) {
	return make_quaternion(q.q0 / c, q.q1 / c, q.q2 / c, q.q3 / c);
};

HOST_DEVICE inline quaternion normalize(const quaternion& q) {
	double length = sqrt(q.q0 * q.q0 + q.q1 * q.q1 + q.q2 * q.q2 + q.q3 * q.q3);
	if (length > 0) return q / length;
	else return q;
};

HOST_DEVICE inline quaternion quaternionRotate(const quaternion& q, const double3& angularVelocity, double timeStep)
{
	double3 v = angularVelocity * timeStep;
	quaternion deltaQ = 0.5 * make_quaternion(-q.q1 * v.x - q.q2 * v.y - q.q3 * v.z,
		q.q0 * v.x + q.q3 * v.y - q.q2 * v.z,
		-q.q3 * v.x + q.q0 * v.y + q.q1 * v.z,
		q.q2 * v.x - q.q1 * v.y + q.q0 * v.z);
	quaternion newQ = normalize(q + deltaQ);
	return newQ;
};

HOST_DEVICE inline double3 rotateVectorByQuaternion(const quaternion& q, const double3& v)
{
	double q0 = q.q0, q1 = q.q1, q2 = q.q2, q3 = q.q3;
	double q00 = q0 * q0;
	double q01 = 2 * q0 * q1;
	double q02 = 2 * q0 * q2;
	double q03 = 2 * q0 * q3;
	double q11 = q1 * q1;
	double q12 = 2 * q1 * q2;
	double q13 = 2 * q1 * q3;
	double q22 = q2 * q2;
	double q23 = 2 * q2 * q3;
	double q33 = q3 * q3;
	double R_xx = q00 + q11 - q22 - q33;
	double R_yx = q12 + q03;
	double R_zx = q13 - q02;
	double R_xy = q12 - q03;
	double R_yy = q00 - q11 + q22 - q33;
	double R_zy = q23 + q01;
	double R_xz = q13 + q02;
	double R_yz = q23 - q01;
	double R_zz = q00 - q11 - q22 + q33;
	return make_double3(R_xx * v.x + R_xy * v.y + R_xz * v.z, R_yx * v.x + R_yy * v.y + R_yz * v.z, R_zx * v.x + R_zy * v.y + R_zz * v.z);
}

HOST_DEVICE inline double3 reverseRotateVectorByQuaternion(const double3& v, const quaternion& q)
{
	double q0 = q.q0, q1 = q.q1, q2 = q.q2, q3 = q.q3;
	double q00 = q0 * q0;
	double q01 = 2 * q0 * q1;
	double q02 = 2 * q0 * q2;
	double q03 = 2 * q0 * q3;
	double q11 = q1 * q1;
	double q12 = 2 * q1 * q2;
	double q13 = 2 * q1 * q3;
	double q22 = q2 * q2;
	double q23 = 2 * q2 * q3;
	double q33 = q3 * q3;
	double R_xx = q00 + q11 - q22 - q33;
	double R_yx = q12 + q03;
	double R_zx = q13 - q02;
	double R_xy = q12 - q03;
	double R_yy = q00 - q11 + q22 - q33;
	double R_zy = q23 + q01;
	double R_xz = q13 + q02;
	double R_yz = q23 - q01;
	double R_zz = q00 - q11 - q22 + q33;
	return make_double3(R_xx * v.x + R_yx * v.y + R_zx * v.z, R_xy * v.x + R_yy * v.y + R_zy * v.z, R_xz * v.x + R_yz * v.y + R_zz * v.z);
}