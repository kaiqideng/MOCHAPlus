#include "output.h"

void writeSpheresVTU(const std::string& fileName, const HostSphere& s,
    int   frame,
    double time,
    int   step)
{
    /* --- make sure "output" dir exists (ignore error if it does) --- */
    MKDIR("outputData");

    /* --- build file name: output/spheres_XXXX.vtu --- */
    std::ostringstream fname;
    fname << "outputData/" << fileName << "_" << std::setw(4) << std::setfill('0') << frame << ".vtu";

    std::ofstream out(fname.str().c_str());
    if (!out) throw std::runtime_error("Cannot open " + fname.str());

    out << std::fixed << std::setprecision(7);      // full double precision
    const int N = s.num;

    /* ============ XML HEADER ============ */
    out << "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <UnstructuredGrid>\n";

    /* ---- global FieldData: TIME + STEP ---- */
    out << "    <FieldData>\n"
        "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << time << " </DataArray>\n"
        "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << step << " </DataArray>\n"
        "    </FieldData>\n";

    /* ---- start Piece ---- */
    out << "    <Piece NumberOfPoints=\"" << N
        << "\" NumberOfCells=\"" << N << "\">\n";

    /* ---- Points ---- */
    out << "      <Points>\n"
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        const double3& p = s.state.positions[i];
        out << ' ' << p.x << ' ' << p.y << ' ' << p.z;
    }
    out << "\n        </DataArray>\n"
        "      </Points>\n";

    /* ---- Cells: one VTK_VERTEX per point ---- */
    out << "      <Cells>\n"
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
    out << "\n        </DataArray>\n"
        "      </Cells>\n";

    /* ---- PointData ---- */
    out << "      <PointData Scalars=\"radius\">\n";

    /* helper lambdas replaced by small inline fns (C++03 safe) */
    {   /* scalar double array */
        out << "        <DataArray type=\"Float32\" Name=\"radius\" format=\"ascii\">\n";
        for (size_t i = 0; i < s.radii.size(); ++i) out << ' ' << s.radii[i];
        out << "\n        </DataArray>\n";
    }
    /* vector<double3> helper */
    const struct {
        const char* name;
        const std::vector<double3>& vec;
    } vec3s[] = {
        { "velocity"       , s.state.velocities        },
        { "angularVelocity", s.state.angularVelocities },
        { "force"          , s.state.forces            },
        { "torque"         , s.state.torques           }
    };
    for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
        out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
            << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        const std::vector<double3>& v = vec3s[k].vec;
        for (size_t i = 0; i < v.size(); ++i)
            out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
        out << "\n        </DataArray>\n";
    }
    /* inverseMass scalar */
    out << "        <DataArray type=\"Float32\" Name=\"inverseMass\" format=\"ascii\">\n";
    for (size_t i = 0; i < s.state.inverseMass.size(); ++i)
        out << ' ' << s.state.inverseMass[i];
    out << "\n        </DataArray>\n";

    out << "      </PointData>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n";
}

/*-------------------------------------------------------------
 *  Write one VTU file for contact interactions
 *    interaction : HostBasicInteraction snapshot
 *    frame       : sequential frame counter  (0000,0001,бн)
 *    time        : physical / simulated time (seconds, double)
 *    step        : solver step counter       (int)
 *  Output file   : outputData/interactions_####.vtu
 *------------------------------------------------------------*/
void writeBasicInteractionsVTU(const std::string& fileName, const HostBasicInteraction& inter,
    int   frame,
    double time,
    int   step)
{
    /* 1. create output/ directory if missing */
    MKDIR("outputData");

    /* 2. build file name */
    std::ostringstream fname;
    fname << "outputData/" << fileName << "_"
        << std::setw(4) << std::setfill('0') << frame
        << ".vtu";

    std::ofstream out(fname.str().c_str());
    if (!out) throw std::runtime_error("Cannot open " + fname.str());
    out << std::fixed << std::setprecision(7);

    const int N = inter.num;           /* active interactions */

    /* 3. VTK header + FieldData (TIME & STEP) */
    out <<
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <UnstructuredGrid>\n"
        "    <FieldData>\n"
        "      <DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\"> "
        << time << " </DataArray>\n"
        "      <DataArray type=\"Int32\"   Name=\"STEP\" NumberOfTuples=\"1\" format=\"ascii\"> "
        << step << " </DataArray>\n"
        "    </FieldData>\n"
        "    <Piece NumberOfPoints=\"" << N
        << "\" NumberOfCells=\"" << N << "\">\n";

    /* 4. Points : contactPoint */
    out <<
        "      <Points>\n"
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        const double3& p = inter.contactPoint[i];
        out << ' ' << p.x << ' ' << p.y << ' ' << p.z;
    }
    out <<
        "\n        </DataArray>\n"
        "      </Points>\n";

    /* 5. Cells : one VTK_VERTEX per contact */
    out <<
        "      <Cells>\n"
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << i;
    out <<
        "\n        </DataArray>\n"
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= N; ++i) out << ' ' << i;
    out <<
        "\n        </DataArray>\n"
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << " 1";           /* 1 = VTK_VERTEX */
    out <<
        "\n        </DataArray>\n"
        "      </Cells>\n";

    /* 6. PointData : all per-contact attributes */
    out <<
        "      <PointData Scalars=\"normalOverlap\">\n";

    /* helpers for scalar/vector output (C++03 safe) */
    const std::vector<double>& scalar_overlap = inter.normalOverlap;
    out <<
        "        <DataArray type=\"Float32\" Name=\"normalOverlap\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << scalar_overlap[i];
    out <<
        "\n        </DataArray>\n";

    /* a small helper to dump vector<double3> */
    const struct V3Block {
        const char* name;
        const std::vector<double3>& vec;
    } vecBlocks[] = {
        { "contactNormal" , inter.contactNormal  },
        { "slidingSpring" , inter.slidingSpring  },
        { "rollingSpring" , inter.rollingSpring  },
        { "torsionSpring" , inter.torsionSpring  },
        { "contactForce"  , inter.contactForce   },
        { "contactTorque" , inter.contactTorque  }
    };

    for (size_t k = 0; k < sizeof(vecBlocks) / sizeof(vecBlocks[0]); ++k) {
        out <<
            "        <DataArray type=\"Float32\" Name=\"" << vecBlocks[k].name
            << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        const std::vector<double3>& v = vecBlocks[k].vec;
        for (int i = 0; i < N; ++i)
            out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
        out <<
            "\n        </DataArray>\n";
    }

    out <<
        "      </PointData>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n";
}

void writeBondedInteractionsVTU(const std::string& fileName, const HostBondedInteraction& inter,
    int   frame,
    double time,
    int   step)
{
    /* 1. create output/ directory if missing */
    MKDIR("outputData");

    /* 2. build file name */
    std::ostringstream fname;
    fname << "outputData/" << fileName << "_"
        << std::setw(4) << std::setfill('0') << frame
        << ".vtu";

    std::ofstream out(fname.str().c_str());
    if (!out) throw std::runtime_error("Cannot open " + fname.str());
    out << std::fixed << std::setprecision(7);

    const int N = inter.num;           /* active interactions */

    /* 3. VTK header + FieldData (TIME & STEP) */
    out <<
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <UnstructuredGrid>\n"
        "    <FieldData>\n"
        "      <DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\"> "
        << time << " </DataArray>\n"
        "      <DataArray type=\"Int32\"   Name=\"STEP\" NumberOfTuples=\"1\" format=\"ascii\"> "
        << step << " </DataArray>\n"
        "    </FieldData>\n"
        "    <Piece NumberOfPoints=\"" << N
        << "\" NumberOfCells=\"" << N << "\">\n";

    /* 4. Points : contactPoint */
    out <<
        "      <Points>\n"
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        const double3& p = inter.contactPoint[i];
        out << ' ' << p.x << ' ' << p.y << ' ' << p.z;
    }
    out <<
        "\n        </DataArray>\n"
        "      </Points>\n";

    /* 5. Cells : one VTK_VERTEX per contact */
    out <<
        "      <Cells>\n"
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << i;
    out <<
        "\n        </DataArray>\n"
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= N; ++i) out << ' ' << i;
    out <<
        "\n        </DataArray>\n"
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << " 1";           /* 1 = VTK_VERTEX */
    out <<
        "\n        </DataArray>\n"
        "      </Cells>\n";

    /* 6. PointData : all per-contact attributes */
    out <<
        "      <PointData Scalars=\"bonded\">\n";

    /* helpers for scalar/vector output (C++03 safe) */
    const std::vector<int>& is = inter.isBonded;
    out <<
        "        <DataArray type=\"Int32\" Name=\"bonded\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << is[i];
    out <<
        "\n        </DataArray>\n";

    out <<
        "        <DataArray type=\"Int32\" Name=\"objectPointed\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << inter.objectPointed[i];
    out <<
        "\n        </DataArray>\n"
        "        <DataArray type=\"Int32\" Name=\"objectPointing\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << inter.objectPointing[i];
    out <<
        "\n        </DataArray>\n"
        "        <DataArray type=\"Float32\" Name=\"normalForce\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << inter.normalForce[i];
    out <<
        "\n        </DataArray>\n"
        "        <DataArray type=\"Float32\" Name=\"torsionTorque\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << inter.torsionTorque[i];
    out <<
        "\n        </DataArray>\n";

    /* a small helper to dump vector<double3> */
    const struct V3Block {
        const char* name;
        const std::vector<double3>& vec;
    } vecBlocks[] = {
        { "contactNormal" , inter.contactNormal  },
        { "shearForce"    , inter.shearForce     },
        { "bendingTorque" , inter.bendingTorque  }
    };

    for (size_t k = 0; k < sizeof(vecBlocks) / sizeof(vecBlocks[0]); ++k) {
        out <<
            "        <DataArray type=\"Float32\" Name=\"" << vecBlocks[k].name
            << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        const std::vector<double3>& v = vecBlocks[k].vec;
        for (int i = 0; i < N; ++i)
            out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
        out <<
            "\n        </DataArray>\n";
    }

    out <<
        "      </PointData>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n";
}

void writeTriangleWallVTU(const std::string& fileName, const HostTriangleVertex& vertices,
    const HostTriangleFace& faces,
    const HostDynamicState& wallState,
    int   frame,
    double time,
    int   step)
{
    /* 1. create output/ directory if missing */
    MKDIR("outputData");

    /* 2. file name*/
    std::ostringstream fname;
    fname << "outputData/" << fileName << "_"
        << std::setw(4) << std::setfill('0') << frame
        << ".vtu";

    std::ofstream out(fname.str().c_str());
    if (!out) { fprintf(stderr, "Cannot open %s\n", fname.str().c_str()); return; }
    out << std::fixed << std::setprecision(7);

    const int Nv = vertices.num;
    const int Nf = faces.num;

    /* -------- 3 XML HEADER + FieldData -------- */
    out <<
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <UnstructuredGrid>\n"
        "    <FieldData>\n"
        "      <DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\"> "
        << time << " </DataArray>\n"
        "      <DataArray type=\"Int32\"   Name=\"STEP\" NumberOfTuples=\"1\" format=\"ascii\"> "
        << step << " </DataArray>\n"
        "    </FieldData>\n"
        "    <Piece NumberOfPoints=\"" << Nv
        << "\" NumberOfCells=\"" << Nf << "\">\n";

    /* -------- 4 Points -------- */
    out <<
        "      <Points>\n"
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < Nv; ++i) {
        const double3& p = vertices.positions[i];
        int iw = vertices.vertex2Wall[i];
        double3 pp = rotateVectorByQuaternion(wallState.orientations[iw], p) + wallState.positions[iw];
        out << ' ' << pp.x << ' ' << pp.y << ' ' << pp.z;
    }
    out <<
        "\n        </DataArray>\n"
        "      </Points>\n";

    /* -------- 5 Cells (VTK_TRIANGLE, type=5) -------- */
    out <<
        "      <Cells>\n"
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < Nf; ++i) {
        out << ' ' << faces.vAIndex[i]
            << ' ' << faces.vBIndex[i]
            << ' ' << faces.vCIndex[i];
    }
    out <<
        "\n        </DataArray>\n"
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= Nf; ++i) out << ' ' << 3 * i;
    out <<
        "\n        </DataArray>\n"
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < Nf; ++i) out << " 5";
    out <<
        "\n        </DataArray>\n"
        "      </Cells>\n";

    if (!faces.face2Wall.empty()) {
        out <<
            "      <CellData Scalars=\"face2Wall\">\n"
            "        <DataArray type=\"Int32\" Name=\"face2Wall\" format=\"ascii\">\n";
        for (int i = 0; i < Nf; ++i) out << ' ' << faces.face2Wall[i];
        out <<
            "\n        </DataArray>\n"
            "      </CellData>\n";
    }

    out <<
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n";
}

void writeTriangleWallPressureVTU(const std::string& fileName, const HostBasicInteraction& faceI, const HostBasicInteraction& edgeI, const HostBasicInteraction& vertexI,
    const HostTriangleWall& TW,
    int   frame,
    double time,
    int   step)
{
    /* 1. create output/ directory if missing */
    MKDIR("outputData");

    /* 2. file name*/
    std::ostringstream fname;
    fname << "outputData/" << fileName << "_"
        << std::setw(4) << std::setfill('0') << frame
        << ".vtu";

    std::ofstream out(fname.str().c_str());
    if (!out) { fprintf(stderr, "Cannot open %s\n", fname.str().c_str()); return; }
    out << std::fixed << std::setprecision(7);

    const int Nv = TW.vertex.num;
    const int Nf = TW.face.num;

    /* -------- 3 XML HEADER + FieldData -------- */
    out <<
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <UnstructuredGrid>\n"
        "    <FieldData>\n"
        "      <DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\"> "
        << time << " </DataArray>\n"
        "      <DataArray type=\"Int32\"   Name=\"STEP\" NumberOfTuples=\"1\" format=\"ascii\"> "
        << step << " </DataArray>\n"
        "    </FieldData>\n"
        "    <Piece NumberOfPoints=\"" << Nv
        << "\" NumberOfCells=\"" << Nf << "\">\n";

    /* -------- 4 Points -------- */
    out <<
        "      <Points>\n"
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < Nv; ++i) {
        const double3& p = TW.vertex.positions[i];
        int iw = TW.vertex.vertex2Wall[i];
        double3 pp = rotateVectorByQuaternion(TW.state.orientations[iw], p) + TW.state.positions[iw];
        out << ' ' << pp.x << ' ' << pp.y << ' ' << pp.z;
    }
    out <<
        "\n        </DataArray>\n"
        "      </Points>\n";

    /* -------- 5 Cells (VTK_TRIANGLE, type=5) -------- */
    out <<
        "      <Cells>\n"
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < Nf; ++i) {
        out << ' ' << TW.face.vAIndex[i]
            << ' ' << TW.face.vBIndex[i]
            << ' ' << TW.face.vCIndex[i];
    }
    out <<
        "\n        </DataArray>\n"
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= Nf; ++i) out << ' ' << 3 * i;
    out <<
        "\n        </DataArray>\n"
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < Nf; ++i) out << " 5";
    out <<
        "\n        </DataArray>\n"
        "      </Cells>\n";

    std::vector<double> normalForce(Nf, 0.);
    for (int i = 0; i < faceI.num; ++i)
    {
        int faceId = faceI.objectPointed[i];
        normalForce[faceId] += abs(dot(faceI.contactForce[i], faceI.contactNormal[i]));
    }
    for (int i = 0; i < edgeI.num; ++i)
    {
        int edgeId = edgeI.objectPointed[i];
        int start = edgeId > 0 ? TW.edge.facePrefixSum[edgeId - 1] : TW.edge.facePrefixSum[0];
        int end = TW.edge.facePrefixSum[edgeId];
        for (int j = start;j < end;j++)
        {
            int faceId = TW.edge.edge2Face[j];
            double3 AB = TW.vertex.positions[TW.face.vBIndex[faceId]] - TW.vertex.positions[TW.face.vAIndex[faceId]];
            double3 BC = TW.vertex.positions[TW.face.vCIndex[faceId]] - TW.vertex.positions[TW.face.vBIndex[faceId]];
            double3 faceNormal = normalize(cross(AB, BC));
            normalForce[faceId] += abs(dot(edgeI.contactForce[i], faceNormal)) / double(end - start);
        }
    }
    for (int i = 0; i < vertexI.num; ++i)
    {
        int vertexId = vertexI.objectPointed[i];
        int start = vertexId > 0 ? TW.vertex.facePrefixSum[vertexId - 1] : TW.vertex.facePrefixSum[0];
        int end = TW.vertex.facePrefixSum[vertexId];
        for (int j = start;j < end;j++)
        {
            int faceId = TW.vertex.vertex2Face[j];
            double3 AB = TW.vertex.positions[TW.face.vBIndex[faceId]] - TW.vertex.positions[TW.face.vAIndex[faceId]];
            double3 BC = TW.vertex.positions[TW.face.vCIndex[faceId]] - TW.vertex.positions[TW.face.vBIndex[faceId]];
            double3 faceNormal = normalize(cross(AB, BC));
            normalForce[faceId] += abs(dot(vertexI.contactForce[i], faceNormal)) / double(end - start);
        }
    }

    out <<
        "      <CellData Scalars=\"pressure\">\n"
        "        <DataArray type=\"Float32\" Name=\"pressure\" format=\"ascii\">\n";
    for (int i = 0; i < Nf; ++i)
    {
        double3 AB = TW.vertex.positions[TW.face.vBIndex[i]] - TW.vertex.positions[TW.face.vAIndex[i]];
        double3 BC = TW.vertex.positions[TW.face.vCIndex[i]] - TW.vertex.positions[TW.face.vBIndex[i]];
        double area = length(cross(AB, BC)) / 2.;
        out << ' ' << normalForce[i] / area;
    }
    out <<
        "\n        </DataArray>\n"
        "      </CellData>\n";

    out <<
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n";
}

/*--------------------------------------------------------------
 *  Write one VTU file (ASCII) into ./output/.
 *  file name : outputData/SPHParticles_####.vtu
 *  frame     : frame counter (zero-based)
 *  time      : simulation/physical time
 *  step      : solver step counter
 *-------------------------------------------------------------*/
void writeSPHSpheresVTU(const std::string& fileName, const HostSPH& SPHP, const HostSphere& s,
    int   frame,
    double time,
    int   step)
{
    /* --- make sure "output" dir exists (ignore error if it does) --- */
    MKDIR("outputData");

    /* --- build file name: output/spheres_XXXX.vtu --- */
    std::ostringstream fname;
    fname << "outputData/" << fileName << "_" << std::setw(4) << std::setfill('0') << frame << ".vtu";

    std::ofstream out(fname.str().c_str());
    if (!out) throw std::runtime_error("Cannot open " + fname.str());

    out << std::fixed << std::setprecision(7);      // full double precision
    const int N = SPHP.num;

    /* ============ XML HEADER ============ */
    out << "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <UnstructuredGrid>\n";

    /* ---- global FieldData: TIME + STEP ---- */
    out << "    <FieldData>\n"
        "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << time << " </DataArray>\n"
        "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << step << " </DataArray>\n"
        "    </FieldData>\n";

    /* ---- start Piece ---- */
    out << "    <Piece NumberOfPoints=\"" << N
        << "\" NumberOfCells=\"" << N << "\">\n";

    /* ---- Points ---- */
    out << "      <Points>\n"
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < s.num; ++i) {
        if (s.SPHIndex[i] < 0) continue;
        const double3& p = s.state.positions[i];
        out << ' ' << p.x << ' ' << p.y << ' ' << p.z;
    }
    out << "\n        </DataArray>\n"
        "      </Points>\n";

    /* ---- Cells: one VTK_VERTEX per point ---- */
    out << "      <Cells>\n"
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
    out << "\n        </DataArray>\n"
        "      </Cells>\n";

    /* ---- PointData ---- */
    out << "      <PointData Scalars=\"radius\">\n";

    /* helper lambdas replaced by small inline fns (C++03 safe) */
    {   /* scalar double array */
        out << "        <DataArray type=\"Float32\" Name=\"radius\" format=\"ascii\">\n";
        for (size_t i = 0; i < s.radii.size(); ++i) {
            if (s.SPHIndex[i] < 0) continue;
            out << ' ' << s.radii[i];
        }
        out << "\n        </DataArray>\n";
    }
    /* vector<double3> helper */
    const struct {
        const char* name;
        const std::vector<double3>& vec;
    } vec3s[] = {
        { "velocity"       , s.state.velocities        }
    };
    for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
        out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
            << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        const std::vector<double3>& v = vec3s[k].vec;
        for (size_t i = 0; i < v.size(); ++i) {
            if (s.SPHIndex[i] < 0) continue;
            out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
        }
        out << "\n        </DataArray>\n";
    }

    out << "        <DataArray type=\"Float32\" Name=\"density\" format=\"ascii\">\n";
    for (size_t i = 0; i < N; ++i) {
        out << ' ' << SPHP.density[i];
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"Float32\" Name=\"pressure\" format=\"ascii\">\n";
    for (size_t i = 0; i < N; ++i) {
        out << ' ' << SPHP.pressure[i];
    }
    out << "\n        </DataArray>\n";

    out << "      </PointData>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n";
}

/*--------------------------------------------------------------
 *  Write one VTU file (ASCII) into ./output/.
 *  file name : outputData/solidParticles_####.vtu
 *  frame     : frame counter (zero-based)
 *  time      : simulation/physical time
 *  step      : solver step counter
 *-------------------------------------------------------------*/
void writeSolidSpheresVTU(const std::string& fileName, const HostSphere& s,
    int   frame,
    double time,
    int   step)
{
    /* --- make sure "output" dir exists (ignore error if it does) --- */
    MKDIR("outputData");

    /* --- build file name: output/spheres_XXXX.vtu --- */
    std::ostringstream fname;
    fname << "outputData/" << fileName << "_" << std::setw(4) << std::setfill('0') << frame << ".vtu";

    std::ofstream out(fname.str().c_str());
    if (!out) throw std::runtime_error("Cannot open " + fname.str());

    out << std::fixed << std::setprecision(7);      // full double precision
    int N = 0;
    std::vector<int> clusterId, clumpId;
    std::vector<double> rad;
    std::vector<double3> pos, vel, ang;
    for (int i = 0; i < s.num; ++i) {
        if (s.SPHIndex[i] < 0)
        {
            clusterId.push_back(s.bondClusterIndex[i]);
            clumpId.push_back(s.clumpIndex[i]);
            rad.push_back(s.radii[i]);
            pos.push_back(s.state.positions[i]);
            vel.push_back(s.state.velocities[i]);
            ang.push_back(s.state.angularVelocities[i]);
            N++;
        }
    }

    /* ============ XML HEADER ============ */
    out << "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <UnstructuredGrid>\n";

    /* ---- global FieldData: TIME + STEP ---- */
    out << "    <FieldData>\n"
        "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << time << " </DataArray>\n"
        "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << step << " </DataArray>\n"
        "    </FieldData>\n";

    /* ---- start Piece ---- */
    out << "    <Piece NumberOfPoints=\"" << N
        << "\" NumberOfCells=\"" << N << "\">\n";

    /* ---- Points ---- */
    out << "      <Points>\n"
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << ' ' << pos[i].x << ' ' << pos[i].y << ' ' << pos[i].z;
    }
    out << "\n        </DataArray>\n"
        "      </Points>\n";

    /* ---- Cells: one VTK_VERTEX per point ---- */
    out << "      <Cells>\n"
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
    out << "\n        </DataArray>\n"
        "      </Cells>\n";

    /* ---- PointData ---- */
    out << "      <PointData Scalars=\"radius\">\n";

    /* helper lambdas replaced by small inline fns (C++03 safe) */
    /* scalar double array */
    out << "        <DataArray type=\"Float32\" Name=\"radius\" format=\"ascii\">\n";
    for (size_t i = 0; i < N; ++i) {
        out << ' ' << rad[i];
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"cluster index\" format=\"ascii\">\n";
    for (size_t i = 0; i < N; ++i) {
        out << ' ' << clusterId[i];
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"clump index\" format=\"ascii\">\n";
    for (size_t i = 0; i < N; ++i) {
        out << ' ' << clumpId[i];
    }
    out << "\n        </DataArray>\n";

    /* vector<double3> helper */
    const struct {
        const char* name;
        const std::vector<double3>& vec;
    } vec3s[] = {
        { "velocity"       , vel },
        { "angularVelocity", ang }
    };
    for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
        out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
            << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        const std::vector<double3>& v = vec3s[k].vec;
        for (size_t i = 0; i < N; ++i) {
            out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
        }
        out << "\n        </DataArray>\n";
    }

    out << "      </PointData>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n";
}

void writeBoxSurfaceVTU(const std::string& fileName,
    const double3& minCorner,
    const double3& maxCorner) {
    // 8 corner points
    std::vector<double3> points = {
        {minCorner.x, minCorner.y, minCorner.z}, // 0
        {maxCorner.x, minCorner.y, minCorner.z}, // 1
        {minCorner.x, maxCorner.y, minCorner.z}, // 2
        {maxCorner.x, maxCorner.y, minCorner.z}, // 3
        {minCorner.x, minCorner.y, maxCorner.z}, // 4
        {maxCorner.x, minCorner.y, maxCorner.z}, // 5
        {minCorner.x, maxCorner.y, maxCorner.z}, // 6
        {maxCorner.x, maxCorner.y, maxCorner.z}  // 7
    };

    // 6 faces as quads (4 points each)
    std::vector<std::vector<int>> faces = {
        {0, 1, 3, 2}, // bottom
        {4, 5, 7, 6}, // top
        {0, 1, 5, 4}, // front
        {2, 3, 7, 6}, // back
        {0, 2, 6, 4}, // left
        {1, 3, 7, 5}  // right
    };
    /* --- make sure "output" dir exists (ignore error if it does) --- */
    MKDIR("outputData");

    /* --- build file name: output/spheres_XXXX.vtu --- */
    std::ostringstream fname;
    fname << "outputData/" << fileName << ".vtu";

    std::ofstream out(fname.str().c_str());
    if (!out) throw std::runtime_error("Cannot open " + fname.str());

    out << std::fixed << std::setprecision(6);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "<UnstructuredGrid>\n";
    out << "<Piece NumberOfPoints=\"" << points.size() << "\" NumberOfCells=\"" << faces.size() << "\">\n";

    // Points
    out << "<Points>\n";
    out << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& p : points)
        out << p.x << " " << p.y << " " << p.z << "\n";
    out << "</DataArray>\n</Points>\n";

    // Faces (as quads)
    out << "<Cells>\n";

    // connectivity
    out << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const auto& f : faces)
        for (int id : f)
            out << id << " ";
    out << "\n</DataArray>\n";

    // offsets
    out << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    int offset = 0;
    for (size_t i = 0; i < faces.size(); ++i) {
        offset += 4;
        out << offset << " ";
    }
    out << "\n</DataArray>\n";

    // cell types: 9 = quad
    out << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t i = 0; i < faces.size(); ++i)
        out << "9 ";
    out << "\n</DataArray>\n";

    out << "</Cells>\n";

    // No point/cell data
    out << "<PointData>\n</PointData>\n";
    out << "<CellData>\n</CellData>\n";

    out << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
    out.close();
}

void writeHostDynamicStateToDat(
    const HostDynamicState& state,
    const std::string& filename,
    double time)
{
    MKDIR("outputData");
    if (time == 0) int nDat = removeDatFiles("outputData");

    std::string finalName;
    if (filename.empty()) {
        finalName = "outputData/state.dat";
    }
    else {
        finalName = "outputData/" + filename + ".dat";
    }

    std::ofstream outFile(finalName, std::ios::app);

    if (!outFile) {
        std::cerr << "Cannot open file: " << finalName << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(10);
    size_t n = state.positions.size();

    for (size_t i = 0; i < n; ++i) {
        const double3& pos = state.positions[i];
        const quaternion& ori = state.orientations[i];
        const double3& vel = state.velocities[i];
        const double3& angVel = state.angularVelocities[i];
        const double3& force = state.forces[i];
        const double3& torque = state.torques[i];

        outFile << time << " "
            << i << " "
            << pos.x << " " << pos.y << " " << pos.z << " "
            << ori.q0 << " " << ori.q1 << " " << ori.q2 << " " << ori.q3 << " "
            << vel.x << " " << vel.y << " " << vel.z << " "
            << angVel.x << " " << angVel.y << " " << angVel.z << " "
            << force.x << " " << force.y << " " << force.z << " "
            << torque.x << " " << torque.y << " " << torque.z << "\n";
    }

    outFile.close();
}